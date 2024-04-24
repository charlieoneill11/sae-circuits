import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import fancy_einsum
from tqdm import tqdm
import re
from sklearn.metrics import roc_curve, auc
import transformer_lens.utils as utils
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from sparse_autoencoder import SparseAutoencoder



### FUNCTION DEFINITIONS ###

# Loss function is MSE (reconstruction loss) + L1 norm of the learned activations + similarity loss
def loss_fn(decoded_activations, learned_activations, resid_streams, resid_labels, lambda_=0.01, alpha_=0.5, verbose=False):

    # RECONSTRUCTION LOSS
    recon_loss = F.mse_loss(decoded_activations, resid_streams)

    # SPARSITY LOSS
    learned_activations_flat = einops.rearrange(learned_activations, 'b s n -> (b s) n')
    sparsity_loss = torch.mean(torch.norm(learned_activations_flat, p=1, dim=1))

    # SIMILARITY LOSS
    # Pos and neg - pos is where resid_labels == 1, neg is where resid_labels == 0
    if alpha_ > 0:
        learned_activations_pos = learned_activations[resid_labels == 1, :, :]
        learned_activations_neg = learned_activations[resid_labels == 0, :, :]
        # Currently (N, S, D) and (M, S, D) -> need to be (D, S, N) and (D, S, M)
        learned_activations_pos = einops.rearrange(learned_activations_pos, 'n s d -> d s n')
        learned_activations_neg = einops.rearrange(learned_activations_neg, 'n s d -> d s n')
        pos_sim_loss = calculate_similarity_loss(learned_activations_pos, learned_activations_neg, verbose=verbose)
    else: 
        pos_sim_loss = torch.tensor(0.0)

    # combine
    return recon_loss + (lambda_ * sparsity_loss) + (alpha_ * pos_sim_loss), recon_loss, sparsity_loss, pos_sim_loss


def train(model, n_epochs, optimizer, train_streams, eval_streams, lambda_=0.01, alpha_=0.5, verbose=False):
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        learned_activations, decoded_activations = model(train_streams)
        loss, recon_loss, sparsity_loss, pos_sim_loss = loss_fn(decoded_activations, learned_activations, train_streams, 
                                                                train_labels, lambda_=lambda_, alpha_=alpha_)
        loss.backward()
        optimizer.step()
        if epoch % (n_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                eval_learned_activations, eval_decoded_activations = model(eval_streams)
                eval_loss, _, _, eval_pos_sim_loss = loss_fn(eval_decoded_activations, eval_learned_activations,
                                                             eval_streams, eval_labels, lambda_=lambda_, alpha_=alpha_, verbose=verbose)
                #print(f"Train loss = {loss.item():.4f}, Eval loss = {eval_loss.item():.4f}")
    return model

def feature_string_to_head_and_layer(feature_index, head_labels):

    extraction = head_labels[feature_index]

    if 'mlp' in extraction.lower(): 
        layer = int(extraction.split('_')[0])
        head = 12
        return layer, head

    # Get head and layer e.g. 'L0H1' -> (0, 1)
    # Layer is everything after L and before H
    layer = int(re.findall(r'L(\d+)H', extraction)[0])
    # Head is everything after H
    head = int(re.findall(r'H(\d+)', extraction)[0])

    return layer, head

def gen_array_template(head_labels):

    # Plot the ground truth (head, layer) pairs (1 if in ground truth, 0 otherwise)
    heads = []
    layers = []
    for i, l in enumerate(head_labels):
        layer, head = feature_string_to_head_and_layer(i, head_labels)
        heads.append(head)
        layers.append(layer)

    heads = list(set(heads))
    layers = list(set(layers))

    return np.zeros((len(layers), len(heads)))
    

def gen_softmaxed_unique_to_pos(all_indices, ground_truth_array, head_labels, normalise=False, num_codes=250):
    # Negative and positive indices
    halfway = all_indices.shape[0] // 2
    positive_indices = all_indices[halfway-num_codes:halfway, :]
    negative_indices = all_indices[halfway:halfway+num_codes, :]

    unique_to_positive_array = gen_array_template(head_labels)
    unique_to_negative_array = gen_array_template(head_labels)

    for i in range(len(head_labels)):
        # Calculate head and layer
        layer, head = feature_string_to_head_and_layer(i, head_labels)

        positive = set(positive_indices[:, i].tolist())
        negative = set(negative_indices[:, i].tolist())
        total_unique = positive.union(negative)

        # In positive but not negative
        unique_to_positive = list(positive - negative)
        # In negative but not positive
        unique_to_negative = list(negative - positive)

        if normalise:
            # Normalise by total number of unique indices
            unique_to_positive_array[layer, head] = len(unique_to_positive) / len(total_unique)
            unique_to_negative_array[layer, head] = len(unique_to_negative) / len(total_unique)
        
        else:
            # Set the values
            unique_to_positive_array[layer, head] = len(unique_to_positive)
            unique_to_negative_array[layer, head] = len(unique_to_negative)

    # Plot the ROC curve in plotly
    y_true = ground_truth_array.flatten()
    y_pred = unique_to_positive_array.flatten()

    # Normalise y_pred with softmax
    def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=0)
    y_pred = softmax(y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Calculate ROC AUC
    roc_auc = auc(fpr, tpr)

    # Calculate F1
    f1 = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))

    return y_true, y_pred, fpr, tpr, roc_auc, f1, thresholds

def gen_co_occurrence_matrix(all_indices, n_heads, n_feat):
    co_occurrence_matrix = np.zeros((n_heads, n_heads, n_feat, n_feat))

    for e in range(all_indices.shape[0]):  # For each example
        for h1 in range(n_heads):  # For each head
            c1 = all_indices[e, h1]  # Code in head h1
            for h2 in range(n_heads):  # For each other head
                if h1 != h2:  # Skip counting co-occurrence of a head with itself
                    c2 = all_indices[e, h2]  # Code in head h2
                    # Increment co-occurrence count for (h1, h2)
                    co_occurrence_matrix[h1, h2, c1, c2] += 1

    return co_occurrence_matrix

def normalize_co_occurrence_matrix(co_occurrence_matrix):
    # Assuming co_occurrence_matrix is of shape (n_heads, n_heads, n_feat, n_feat)
    n_heads, _, n_feat, _ = co_occurrence_matrix.shape
    normalized_matrix = np.zeros_like(co_occurrence_matrix)

    for h1 in range(n_heads):
        for h2 in range(n_heads):
            if h1 != h2:  # Skip self co-occurrences
                total_co_occurrences = np.sum(co_occurrence_matrix[h1, h2, :, :])
                if total_co_occurrences > 0:  # Avoid division by zero
                    normalized_matrix[h1, h2, :, :] = co_occurrence_matrix[h1, h2, :, :] / total_co_occurrences

    return normalized_matrix

def unique_co_occurrences(positive_matrix, negative_matrix, normalise=True):
    # Normalize matrices
    if normalise:
        positive_matrix = normalize_co_occurrence_matrix(positive_matrix)
        negative_matrix = normalize_co_occurrence_matrix(negative_matrix)

    n_heads, _, n_feat, _ = positive_matrix.shape
    unique_co_occurrence_counts = np.zeros((n_heads, n_heads))
    
    for h1 in range(n_heads):
        for h2 in range(n_heads):
            if h1 != h2:  # Skip self co-occurrences
                # Find co-occurrences in positive not present in negative
                unique_positives = positive_matrix[h1, h2, :, :] > 0
                negatives = negative_matrix[h1, h2, :, :] > 0
                # Boolean array of unique positives
                unique = unique_positives & ~negatives
                if normalise:
                    # Normalize count by total co-occurrences for this head pair in positive matrix
                    total_co_occurrences = np.sum(positive_matrix[h1, h2, :, :] > 0) + np.sum(negative_matrix[h1, h2, :, :] > 0)
                    if total_co_occurrences > 0:  # Avoid division by zero
                        unique_count_normalized = np.sum(unique) / total_co_occurrences
                    else:
                        unique_count_normalized = 0
                    # Set normalized unique counts for this head pair
                    unique_co_occurrence_counts[h1, h2] = unique_count_normalized
                else:
                    # Count unique co-occurrences
                    unique_co_occurrence_counts[h1, h2] = np.sum(unique)

    return unique_co_occurrence_counts

def pairwise_cosine_similarities(pos_examples, neg_examples, eps=1e-12):
    """
    pos_examples = (D, S, N)
    neg_examples = (D, S, M)

    Calculate the average cosine similarity for vectors at the same sequence
    position in pos_examples and neg_examples, vectorized.
    """

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

    return final_scalar

def calculate_similarity_loss(pos_examples, neg_examples, eps=1e-12, delta=1.0, verbose=False):

    # Positive-negative
    pos_neg_scalar = pairwise_cosine_similarities(pos_examples, neg_examples, eps)
    if verbose: print(f"Pos-neg loss = {pos_neg_scalar.item():.4f}")

    # Positive-positive
    pos_pos_scalar = pairwise_cosine_similarities(pos_examples, pos_examples, eps)
    if verbose: print(f"Pos-pos loss = {pos_pos_scalar.item():.4f}")
    
    return pos_neg_scalar + (delta - (pos_pos_scalar))

def calculate_f1_score(y_true, y_pred):
    # Calculate True Positives (TP)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate False Positives (FP)
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    # Calculate False Negatives (FN)
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate Treu Negatives (TN)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    # Calculate Precision and Recall
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 Score
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    
    return F1, Precision, Recall, TP, FP, TN, FN

### MAIN CODE ###

task = 'ioi'
task_type = 'node'
assert task_type in ['node', 'edge'], "Type must be either 'node' or 'edge'"
print(f"Type: {task_type}")
task_mappings = {
    'gt': 'Greater-than',
    'ioi': 'Indirect Object Identification',
    'ds': 'Docstring',
    'induction': 'Induction',
    'tracr-reverse': 'Tracr-Reverse'
}

print(f"Task: {task_mappings[task]}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

num_unique = 300
n_epochs = 500
num_trials = 1

roc_results = []


for i in range(num_trials):
    # Load residual streams
    resid_streams = torch.load(f"../data/{task}/resid_heads_mean.pt").to(device)
    head_labels = torch.load(f'../data/{task}/labels_heads_mean.pt')
    ground_truth = torch.load(f'../data/{task}/ground_truth.pt')


    # Shuffle and create the labels
    labels = torch.ones(resid_streams.shape[0]//2) # BIG ASSUMPTION: assumes first half is positive and second half is negative
    labels = torch.cat((labels, torch.zeros_like(labels)))
    permutation = torch.randperm(resid_streams.shape[0])
    resid_shuffled = resid_streams[permutation, :, :]
    labels_shuffled = labels[permutation]
    cutoff = 10 #int(resid_shuffled.shape[0] * 0.8)
    train_streams = resid_shuffled[:cutoff, :, :].to(device)
    train_labels = labels_shuffled[:cutoff].to(device)
    eval_streams = resid_shuffled[cutoff:, :, :].to(device)
    eval_labels = labels_shuffled[cutoff:].to(device)


    model = SparseAutoencoder(n_input_features=resid_streams.shape[-1], n_learned_features=num_unique, geometric_median_dataset=None).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train(model, n_epochs, optimizer, train_streams, eval_streams, lambda_=0.02, alpha_=0.0)
    model = model.to('cpu')
    resid_streams = resid_streams.to('cpu')
    # Save model
    torch.save(model, f'../models/{task}/sparse_autoencoder_{task_type}.pt')

    heads = []
    layers = []
    for i, l in enumerate(head_labels):
        layer, head = feature_string_to_head_and_layer(i, head_labels)
        heads.append(head)
        layers.append(layer)

    heads = list(set(heads))
    layers = list(set(layers))

    ground_truth_array = np.zeros((len(layers), len(heads)))
    for layer, head in ground_truth:
        ground_truth_array[layer, head] = 1

    normalise = False# if task == 'ds' else True

    # Plot the ground truth (head, layer) pairs (1 if in ground truth, 0 otherwise)
    if task_type == 'node':

        model.eval()
        learned_activations = model(resid_streams)[0].detach().cpu().numpy()
        all_indices = np.argmax(learned_activations, axis=2)

        print(f"\n\nNormalise: {normalise}")
        y_true, y_pred, fpr, tpr, node_roc_auc, f1, thresholds = gen_softmaxed_unique_to_pos(all_indices, ground_truth_array, head_labels, normalise=normalise)

        # Print best f1 score (and corresponding threshold)
        node_best_f1 = np.max(f1)
        best_threshold = thresholds[np.argmax(f1)]
        print(f"Best F1 score: {node_best_f1:.4f}")

        # Print ROC AUC
        print(f"ROC AUC: {node_roc_auc:.4f}\n\n")

        roc_results.append(node_roc_auc)

    elif task_type == 'edge':
        def gen_co_occurrence_matrix(all_indices, n_heads, n_feat):
            co_occurrence_matrix = np.zeros((n_heads, n_heads, n_feat, n_feat))

            for e in range(all_indices.shape[0]):  # For each example
                for h1 in range(n_heads):  # For each head
                    c1 = all_indices[e, h1]  # Code in head h1
                    for h2 in range(n_heads):  # For each other head
                        if h1 != h2:  # Skip counting co-occurrence of a head with itself
                            c2 = all_indices[e, h2]  # Code in head h2
                            # Increment co-occurrence count for (h1, h2)
                            co_occurrence_matrix[h1, h2, c1, c2] += 1

            return co_occurrence_matrix

        def normalize_co_occurrence_matrix(co_occurrence_matrix):
            # Assuming co_occurrence_matrix is of shape (n_heads, n_heads, n_feat, n_feat)
            n_heads, _, n_feat, _ = co_occurrence_matrix.shape
            normalized_matrix = np.zeros_like(co_occurrence_matrix)

            for h1 in range(n_heads):
                for h2 in range(n_heads):
                    if h1 != h2:  # Skip self co-occurrences
                        total_co_occurrences = np.sum(co_occurrence_matrix[h1, h2, :, :])
                        if total_co_occurrences > 0:  # Avoid division by zero
                            normalized_matrix[h1, h2, :, :] = co_occurrence_matrix[h1, h2, :, :] / total_co_occurrences

            return normalized_matrix

        def unique_co_occurrences(positive_matrix, negative_matrix, normalise=True):
            # Normalize matrices
            if normalise:
                positive_matrix = normalize_co_occurrence_matrix(positive_matrix)
                negative_matrix = normalize_co_occurrence_matrix(negative_matrix)

            n_heads, _, n_feat, _ = positive_matrix.shape
            unique_co_occurrence_counts = np.zeros((n_heads, n_heads))
            
            for h1 in range(n_heads):
                for h2 in range(n_heads):
                    if h1 != h2:  # Skip self co-occurrences
                        # Find co-occurrences in positive not present in negative
                        unique_positives = positive_matrix[h1, h2, :, :] > 0
                        negatives = negative_matrix[h1, h2, :, :] > 0
                        # Boolean array of unique positives
                        unique = unique_positives & ~negatives
                        if normalise:
                            # Normalize count by total co-occurrences for this head pair in positive matrix
                            total_co_occurrences = np.sum(positive_matrix[h1, h2, :, :] > 0) + np.sum(negative_matrix[h1, h2, :, :] > 0)
                            if total_co_occurrences > 0:  # Avoid division by zero
                                unique_count_normalized = np.sum(unique) / total_co_occurrences
                            else:
                                unique_count_normalized = 0
                            # Set normalized unique counts for this head pair
                            unique_co_occurrence_counts[h1, h2] = unique_count_normalized
                        else:
                            # Count unique co-occurrences
                            unique_co_occurrence_counts[h1, h2] = np.sum(unique)

            return unique_co_occurrence_counts

        # Learned activations and then take argmax to discretise
        learned_activations = model(resid_streams)[0].detach().cpu().numpy()
        all_indices = np.argmax(learned_activations, axis=2)

        positive_indices = all_indices[:250, :]
        negative_indices = all_indices[250:, :]
        positive_learned_activations = learned_activations[:250, :, :]
        negative_learned_activations = learned_activations[250:, :, :]

        # Assume all_indices, positive_indices, and negative_indices are defined, as well as n_heads and n_feat
        n_feat = learned_activations.shape[-1]
        n_heads = all_indices.shape[1]
        positive_co_occurrence_matrix = gen_co_occurrence_matrix(positive_indices, n_heads, n_feat)
        negative_co_occurrence_matrix = gen_co_occurrence_matrix(negative_indices, n_heads, n_feat)

        # Calculate unique co-occurrences
        normalise = False# if task in ['ds', 'ioi']  else True
        print(f"Normalise: {normalise}")
        unique_co_occurrence_counts = unique_co_occurrences(positive_co_occurrence_matrix, negative_co_occurrence_matrix, normalise=normalise)

        # Sort (head, head) pairs by descending unique co-occurrence counts
        sorted_indices = np.argsort(unique_co_occurrence_counts.flatten())[::-1]
        sorted_indices = np.unravel_index(sorted_indices, unique_co_occurrence_counts.shape)
        # Zip them together to create a list of (head, head) pairs
        sorted_head_pairs = list(zip(sorted_indices[0], sorted_indices[1]))

        circuit_components = []
        for i, (h1, h2) in enumerate(sorted_head_pairs):
            (l1, h1) = feature_string_to_head_and_layer(h1, head_labels)
            (l2, h2) = feature_string_to_head_and_layer(h2, head_labels)
            circuit_components.append((l1, h1))
            circuit_components.append((l2, h2))

        if len(circuit_components) > 10000:
            k = 10000
        else:
            k = 750

        y_pred = np.zeros_like(ground_truth_array)
        for (l, h) in circuit_components[:k]:
            y_pred[l, h] += 1

        # Plot the ROC curve in plotly
        y_true = ground_truth_array.flatten()
        y_pred = y_pred.flatten()

        # Normalise y_pred with softmax
        def softmax_edge(x): return np.exp(x) / np.sum(np.exp(x), axis=0)
        y_pred = softmax_edge(y_pred)

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Calculate ROC AUC
        roc_auc = auc(fpr, tpr)

        # Calculate F1
        f1 = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))

        # Print
        print(f"ROC AUC: {roc_auc:.4f}")
        best_f1 = np.max(f1)
        print(f"Best F1 score: {best_f1:.4f}")
        # roc_results.append(roc_auc)

        # For varying values of k, calculate ROC AUC and F1 score
        if len(circuit_components) > 10000:
            k_values = np.arange(1, len(circuit_components), 100)
        else:
            k_values = np.arange(1, len(circuit_components), 10)
        roc_auc_values = []
        f1_values = []

        for k in k_values:
            y_pred = np.zeros_like(ground_truth_array)
            for (l, h) in circuit_components[:k]:
                y_pred[l, h] += 1

            y_true = ground_truth_array.flatten()
            y_pred = y_pred.flatten()

            y_pred = softmax_edge(y_pred)

            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            roc_auc_values.append(roc_auc)

            f1 = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))
            f1_values.append(np.max(f1))

        print(f"Best F1 score across k: {np.max(f1_values):.4f}")
        print(f"Best ROC AUC across k: {np.max(roc_auc_values):.4f} (at k={k_values[np.argmax(roc_auc_values)]})")
        roc_results.append(np.max(roc_auc_values))


    else:
        raise ValueError("Task type must be either 'node' or 'edge'")


print(roc_results)
print(f"Mean ROC = {np.mean(roc_results):.4f} (+/- {np.std(roc_results):.4f})")