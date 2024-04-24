import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import einops
from fancy_einsum import einsum
from tqdm import tqdm
import argparse
from datetime import datetime
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import yaml
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import re


### OUR IMPORTS ###
from model import TransformerAutoencoder, TransformerVQVAE
from common_utils import load_config

### OUR FUNCTIONS ###
def calculate_positional_similarity_loss(pos_examples, neg_examples):
    """
    pos_examples = (D, S, N)
    neg_examples = (D, S, M)

    Calculate the average cosine similarity for vectors at the same sequence
    position in pos_examples and neg_examples, vectorized.
    """
    # Initialize epsilon for numerical stability in divisions
    eps = 1e-12

    # Reshape tensors for dot product computation
    pos_examples_perm = pos_examples.permute(1, 2, 0)  # Change to shape (S, N, D) for batch processing
    neg_examples_perm = neg_examples.permute(1, 0, 2)  # Change to shape (S, D, M) for correct dot product

    # Compute dot products. Now, using einsum for clarity and correctness
    dot_products = torch.einsum('snd,sdm->snm', pos_examples_perm, neg_examples_perm)

    # Calculate magnitudes for normalization
    magnitude_p = torch.sqrt(torch.einsum('snd,snd->sn', pos_examples_perm, pos_examples_perm) + eps).unsqueeze(-1)
    magnitude_n = torch.sqrt(torch.einsum('sdm,sdm->sm', neg_examples_perm, neg_examples_perm) + eps).unsqueeze(-2)

    # Calculate cosine similarities
    cosine_similarities = dot_products / (magnitude_p * magnitude_n + eps)

    # Average the cosine similarities for each position across all N, M pairs
    average_cosine_similarities_per_position = torch.mean(cosine_similarities, dim=(1, 2))

    # Finally, average these across all sequence positions
    final_scalar = torch.mean(average_cosine_similarities_per_position)

    return final_scalar, average_cosine_similarities_per_position  

def head_index_to_head_and_layer(index, head_labels):
    return head_labels[index]

def feature_string_to_head_and_layer(feature_index, head_labels):

    extraction = head_index_to_head_and_layer(feature_index, head_labels)

    # Get head and layer e.g. 'L0H1' -> (0, 1)
    # Layer is everything after L and before H
    layer = int(re.findall(r'L(\d+)H', extraction)[0])
    # Head is everything after H
    head = int(re.findall(r'H(\d+)', extraction)[0])

    return layer, head

def feature_index_to_index_cutoff(feature_index, head_labels):
    layer, head = feature_string_to_head_and_layer(feature_index, head_labels)
    # Cut off all the layers after and including this one
    new_index = -1
    for i, label in enumerate(head_labels):
        label_layer = int(re.findall(r'L(\d+)H', label)[0])
        if label_layer < layer:
            new_index = i
        else: break
    return new_index+1

def get_all_indices(model, resid_streams):
    model.eval()
    _, all_indices = model.quantized_indices(resid_streams.cpu())

    # all_quantised, all_indices = model.quantized_indices(resid_streams.cpu())
    # all_quantised = all_quantised.detach().cpu().numpy()
    all_indices = all_indices.T

    # Step 1: Identify the unique integers in the tensor
    unique_integers = torch.unique(all_indices)

    # Step 2: Sort these unique integers
    sorted_unique_integers = torch.sort(unique_integers).values

    # Step 3: Create a mapping from original integers to new integers (0-17)
    mapping = {old_val.item(): new_val for new_val, old_val in enumerate(sorted_unique_integers)}

    # Step 4: Apply the mapping to the original tensor
    mapped_tensor = all_indices.clone()  # To avoid modifying the original tensor
    for old_val, new_val in mapping.items():
        mapped_tensor[all_indices == old_val] = new_val

    return mapped_tensor, len(sorted_unique_integers)

def circuit_f1(feature_importance_circuit, ground_truth):

    # Convert feature importance circuit to a set of tuples
    feature_tuples = []
    for idx, (layer, head, code) in enumerate(feature_importance_circuit):
        feature_tuples.append((layer, head))

    # Calculate precision and recall between tuples
    tp = len(set(feature_tuples).intersection(set(ground_truth)))
    fp = len(set(feature_tuples).difference(set(ground_truth)))
    fn = len(set(ground_truth).difference(set(feature_tuples)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    try: f1 = 2 * (precision * recall) / (precision + recall)
    except: f1 = 0

    return f1, precision, recall

def branch_and_analyze_features(X, y, encoder, head_labels, current_layer, feature_importance_circuit, top_k = 2, verbose=True):
    if current_layer <= 0:
        return feature_importance_circuit

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X, y)
    
    importances = tree.feature_importances_
    top_two_indices = np.argsort(importances)[-top_k:]  # Get indices of top k features
    
    for index in top_two_indices:
        #filtered_importances = importances[:sum(len(c) for c in encoder.categories_[:index+1])]
        most_important_feature = index #np.argmax(filtered_importances)

        # Finding original feature and code

        cumulative_lengths = np.cumsum([len(c) for c in encoder.categories_])
        original_feature = np.searchsorted(cumulative_lengths, most_important_feature, side='right')
        original_code = encoder.categories_[original_feature][most_important_feature - sum([len(c) for c in encoder.categories_[:original_feature]])]

        layer, head = feature_string_to_head_and_layer(original_feature, head_labels)
        feature_importance_circuit.append((layer, head, original_code))

        new_index = feature_index_to_index_cutoff(original_feature, head_labels)
        encoder_cutoff_index = sum([len(c) for c in encoder.categories_[:new_index]])

        # Update targets and data for recursion
        new_target = (X[:, original_feature] == original_code).astype(int)
        X_new = X[:, :encoder_cutoff_index]

        # Recursive call
        branch_and_analyze_features(X_new, new_target, encoder, head_labels, layer-1, feature_importance_circuit, verbose)

    return feature_importance_circuit

### MAIN SCRIPT ###

saved_cfg = torch.load('../models/long_training_2024-03-20_12-53.pt', map_location=torch.device('cpu') )
config = saved_cfg['config']

# Extract residual streams from dataset
resid_streams = torch.load(config['paths']['data'])
# Rearrange with einops
resid_streams = einops.rearrange(resid_streams, 'n l d -> l n d')
device = 'cpu'


# Shuffle and create the labels
labels = torch.ones(resid_streams.shape[1]//2) # BIG ASSUMPTION: assumes first half is positive and second half is negative
labels = torch.cat((labels, torch.zeros_like(labels)))
permutation = torch.randperm(resid_streams.shape[1])
resid_shuffled = resid_streams[:, permutation, :]
labels_shuffled = labels[permutation]
cutoff = int(resid_shuffled.shape[1] * 0.8)
train_streams = resid_shuffled[:, :cutoff, :].to(device)
train_labels = labels_shuffled[:cutoff].to(device)
eval_streams = resid_shuffled[:, cutoff:, :].to(device)
eval_labels = labels_shuffled[cutoff:].to(device)

config = saved_cfg['config']
epochs = config['training']['epochs']
print_epoch = config['training']['print_epoch']
alpha = config['training']['alpha']
verbose = config['training']['verbose']
date_hour_minute_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Hyperparameters
input_dim = resid_streams.shape[-1]
d_model = config['model']['d_model']
nhead = config['model']['nhead']
num_encoder_layers = config['model']['num_encoder_layers']
num_decoder_layers = config['model']['num_decoder_layers']
dim_feedforward = config['model']['dim_feedforward']
dropout = config['model']['dropout']

# Instantiate model
if config['model']['type'] == 'fsq':
    model = TransformerAutoencoder(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=dropout).to(device)
elif config['model']['type'] == 'vq':
    codebook_size = config['model']['codebook_size']
    codebook_dim = config['model']['codebook_dim']
    threshold_ema_dead_code = config['model']['threshold_ema_dead_code']
    model = TransformerVQVAE(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=dropout, codebook_size=codebook_size, codebook_dim=codebook_dim, threshold_ema_dead_code=threshold_ema_dead_code).to(device)

# Load
model.load_state_dict(saved_cfg['model_state_dict'])


### SIMILARITY LOSS ###
model.eval()
positive_indices = torch.where(train_labels == 1.0)[0]
negative_indices = torch.where(train_labels == 0.0)[0]
pos_streams = train_streams[:, positive_indices, :]
neg_streams = train_streams[:, negative_indices, :]
pos_quant = model.encode(pos_streams)
neg_quant = model.encode(neg_streams)
pos_quant = einops.rearrange(pos_quant, 'S N D -> D S N')
neg_quant = einops.rearrange(neg_quant, 'S N D -> D S N')
sim_loss, average_cosine_sim = calculate_positional_similarity_loss(pos_quant, neg_quant)
print(f"Similarity between positive and negative indices: {sim_loss.item():.4f}")

pos_sim_loss, pos_average_cosine_sim = calculate_positional_similarity_loss(pos_quant, pos_quant)
neg_sim_loss, neg_average_cosine_sim = calculate_positional_similarity_loss(neg_quant, neg_quant)
print(f"Similarity between positive indices: {pos_sim_loss.item():.4f}")
print(f"Similarity between negative indices: {neg_sim_loss.item():.4f}\n\n")


### SEQUENTIAL RECURSIVE DT CIRCUIT FINDER ###

resid_streams = torch.load(config['paths']['data'])
resid_streams = einops.rearrange(resid_streams, 'n l d -> l n d')
head_labels = torch.load('../data/ioi3/head_labels.pt')
all_indices, num_unique = get_all_indices(model, resid_streams)

# Initial labels setup
halfway = all_indices.shape[0] // 2
labels = np.ones(halfway)
labels = np.concatenate((labels, np.zeros(halfway)))

# OneHotEncoder setup and transformation
categories = [np.arange(num_unique) for _ in range(144)]  # assuming 144 features, each with 32 categories
encoder = OneHotEncoder(categories=categories, sparse_output=True)
all_indices_encoded = encoder.fit_transform(all_indices).toarray()

# Initial train-test split
X_train, X_test, y_train, y_test = train_test_split(all_indices_encoded, labels, test_size=1, random_state=42)
feature_importance_circuit = []
current_target_index = all_indices.shape[1]-1  # Starting with the last feature
feature_mapping = []

# Generate feature mapping for interpretability
for i, categories in enumerate(encoder.categories_):
    layer, head = feature_string_to_head_and_layer(feature_index=i, head_labels=head_labels)
    new_features = [f"Layer_{layer}_Head_{head}_Code_{category}" for category in categories]
    feature_mapping.extend(new_features)

counter = 0
while layer > 0:
    counter += 1
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    
    importances = tree.feature_importances_
    filtered_importances = importances[:sum(len(c) for c in encoder.categories_[:current_target_index+1])]
    most_important_feature = np.argmax(filtered_importances)
    
    # Identify the original feature and code
    for i, cat_len in enumerate([len(c) for c in encoder.categories_]):
        if most_important_feature < sum([len(c) for c in encoder.categories_[:i+1]]):
            original_feature = i
            break
    original_code = encoder.categories_[original_feature][most_important_feature - sum([len(c) for c in encoder.categories_[:original_feature]])]
    layer, head = feature_string_to_head_and_layer(feature_index=original_feature, head_labels=head_labels)
    feature_importance_circuit.append((layer, head, original_code))

    # Cutoff
    new_index = feature_index_to_index_cutoff(original_feature, head_labels)
    # Map new index back to encoder indices
    encoder_cutoff_index = sum([len(c) for c in encoder.categories_[:new_index]])
    
    # Update target labels based on presence of the identified code
    new_target = (all_indices[:, original_feature] == original_code)
    # Convert new_target from tensor of bools to tensor of ints
    new_target = new_target.int()
    y_train = new_target[:len(y_train)]
    y_test = new_target[len(y_train):]

    # Adjust training and testing data for next iteration
    X_train = X_train[:, :encoder_cutoff_index]
    X_test = X_test[:, :encoder_cutoff_index]
    
    # Prepare for next iteration
    current_target_index = original_feature - 1

# Reverse to start from the first feature for presentation
feature_importance_circuit = feature_importance_circuit[::-1]

# Print the circuit
# print("Circuit according to sequential recursive DT:")
# for idx, (layer, head, code) in enumerate(feature_importance_circuit):
#     print(f"Layer {layer}, Head {head}, Code {code}")

ground_truth = [
    (2, 2), (4, 11), # previous token heads
    (0, 1), (3, 0), (0, 10), # duplicate token heads
    (5, 5), (6, 9), (5, 8), (5, 9), # induction heads
    (7, 3), (7, 9), (8, 6), (8, 10), # S-inhibition heads
    (10, 7), (11, 10), # negative name-mover heads
    (9, 9), (9, 6), (10, 0), # name-mover heads
    (10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (11, 9), (11, 3), (9, 7), # backup name-mover heads
]


f1, precision, recall = circuit_f1(feature_importance_circuit, ground_truth)
print("Results for sequential recursive DT circuit finder:")
print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n\n")


### BRANCHED RECURSIVE DT CIRCUIT FINDER ###
all_indices, num_unique = get_all_indices(model, resid_streams)
halfway = all_indices.shape[0] // 2
labels = np.ones(halfway)
labels = np.concatenate((labels, np.zeros(halfway)))

# OneHotEncoder setup and transformation
categories = [np.arange(num_unique) for _ in range(144)]  # assuming 144 features, each with 32 categories
encoder = OneHotEncoder(categories=categories, sparse_output=True)
all_indices_encoded = encoder.fit_transform(all_indices).toarray()
X_train, X_test, y_train, y_test = train_test_split(all_indices_encoded, labels, test_size=1, random_state=42)
starting_layer = X_train.shape[1] // num_unique // 12 # 12 is number of layers 
feature_importance_circuit = branch_and_analyze_features(X_train, y_train, encoder, head_labels, starting_layer, [], top_k=5)
feature_importance_circuit = list(set(feature_importance_circuit))

# Sort by layer
feature_importance_circuit = sorted(feature_importance_circuit, key=lambda x: x[0])

# Print the circuit
# print("Circuit according to DT:")
# for idx, (layer, head, code) in enumerate(feature_importance_circuit):
#     print(f"Layer {layer}, Head {head}, Code {code}")
# print()


f1, precision, recall = circuit_f1(feature_importance_circuit, ground_truth)
print("Results for branched recursive DT circuit finder:")
print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = []
for k in top_k_list:
    feature_importance_circuit = branch_and_analyze_features(X_train, y_train, encoder, head_labels, starting_layer, [], top_k=k)
    feature_importance_circuit = list(set(feature_importance_circuit))
    feature_importance_circuit = sorted(feature_importance_circuit, key=lambda x: x[0])
    f1, precision, recall = circuit_f1(feature_importance_circuit, ground_truth)
    results.append((f1, precision, recall))

best_f1 = max([r[0] for r in results])
print(f"Best F1 score: {best_f1:.4f} at k = {top_k_list[np.argmax([r[0] for r in results])]}\n\n")

# Plotly plot results against k
fig = go.Figure()
fig.add_trace(go.Scatter(x=top_k_list, y=[r[0] for r in results], mode='lines+markers', name='F1'))
fig.add_trace(go.Scatter(x=top_k_list, y=[r[1] for r in results], mode='lines+markers', name='Precision'))
fig.add_trace(go.Scatter(x=top_k_list, y=[r[2] for r in results], mode='lines+markers', name='Recall'))
fig.update_layout(title='F1, Precision, and Recall against k',
                   xaxis_title='k',
                   yaxis_title='Score')
model_title = config['paths']['data'].split('/')[-1].split('.')[0]
fig.write_image(f"../output/ioi/branch_circuit_finding_k_{model_title}.pdf")
#fig.show()



### DECISION TREE ON QUANTISED INDICES ###
model.eval()
all_indices, num_unique = get_all_indices(model, resid_streams)
halfway = all_indices.shape[0] // 2
labels = np.ones(halfway)
labels = np.concatenate((labels, np.zeros(halfway)))

# OneHotEncoder setup and transformation
categories = [np.arange(num_unique) for _ in range(144)]  # assuming 144 features, each with num_unique categories
encoder = OneHotEncoder(categories=categories, sparse_output=True)
all_indices_encoded = encoder.fit_transform(all_indices).toarray()

# Train test split on the one-hot encoded data
X_train, X_test, y_train, y_test = train_test_split(all_indices_encoded, labels, test_size=0.2, random_state=49)

# Train the decision tree model on the one-hot encoded data
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Get the predictions
predictions = tree.predict(X_test)
dt_indices_accuracy = (predictions == y_test).mean()
print(f"DT accuracy on quantised indices = {dt_indices_accuracy*100:.2f}%\n\n")


### DECISION TREE ON ORIGINAL QUANTISED VECTORS ###

quantised, indices = model.quantized_indices(resid_streams)
quantised = quantised.detach().cpu().numpy()
quantised = einops.rearrange(quantised, 'S N D -> (S D) N')
labels = torch.ones(quantised.shape[1]//2)
labels = torch.cat((labels, torch.zeros_like(labels)))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(quantised.T, labels, test_size=0.2)

# Train a decision tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
dt_original_accuracy = accuracy_score(y_test, y_pred)
print(f"DT accuracy on original quantised vectors = {dt_original_accuracy*100:.2f}%\n\n")
print(f"Difference in accuracy between vectors and indices = {(dt_original_accuracy - dt_indices_accuracy)*100:.2f}%")