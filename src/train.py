"""
This script trains a transformer autoencoder on an arbitrary dataset.
The autoencoder is trained on reconstruction loss and an optional similarity loss.
Saving the model is done when the best evaluation loss is achieved.
The weights are saved along with the config file used for training to the models folder.
The saved weights (as well as the data used for training) should be used for downstream analysis.
"""

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

### OUR IMPORTS ###
from model import TransformerAutoencoder, TransformerVQVAE
from common_utils import load_config


# Similarity loss is a quasi-contrastive loss (without maximising the similarity of examples with the same class)
# def pairwise_cosine_similarities(pos_examples, neg_examples, eps=1e-12):
#     """
#     pos_examples = (D, S, N)
#     neg_examples = (D, S, M)
#     """
#     pos_expanded = pos_examples.unsqueeze(-1) # Now (D, S, N, 1)
#     neg_expanded = neg_examples.unsqueeze(2) # Now (D, S, 1, M)

#     # First, compute dot products and magnitudes for normalisation
#     dot_products = torch.sum(pos_expanded * neg_expanded, dim=0)  # Sum over D dimension, shape becomes (S, N, M)
#     magnitude_p = torch.sqrt(torch.sum(pos_expanded ** 2, dim=0))  # Sum over D, shape (S, N, 1)
#     magnitude_n = torch.sqrt(torch.sum(neg_expanded ** 2, dim=0))  # Sum over D, shape (S, 1, M)

#     # Now calculate cosine similarities and then average over the sequence length (S)
#     cosine_similarities = dot_products / ((magnitude_p * magnitude_n) + eps)  # Broadcasting handles the division properly
#     average_cosine_similarities = torch.mean(cosine_similarities, dim=0)  # Average over S, shape (N, M)

#     # Average over all pairs to get a single scalar
#     final_scalar = torch.mean(average_cosine_similarities)

#     return final_scalar

def calculate_similarity_loss(pos_examples, neg_examples, eps=1e-12, delta=1.0, verbose=False):

    # Positive-negative
    pos_neg_scalar = pairwise_cosine_similarities(pos_examples, neg_examples, eps)
    if verbose: print(f"Pos-neg loss = {pos_neg_scalar.item():.4f}")

    # Positive-positive
    pos_pos_scalar = pairwise_cosine_similarities(pos_examples, pos_examples, eps) / 2
    if verbose: print(f"Pos-pos loss = {pos_pos_scalar.item():.4f}")

    # Negative-negative
    neg_neg_scalar = pairwise_cosine_similarities(neg_examples, neg_examples, eps) / 2
    if verbose: print(f"Neg-neg loss = {neg_neg_scalar.item():.4f}")

    if verbose: print(f"Opposites loss = {max(0, delta - (pos_pos_scalar + neg_neg_scalar))}")
    
    return pos_neg_scalar + (delta - (pos_pos_scalar + neg_neg_scalar))

def pairwise_cosine_similarities(pos_examples, neg_examples, eps = 1e-12):
    """
    pos_examples = (D, S, N)
    neg_examples = (D, S, M)

    For each sequence position, calculate the average cosine similarity
    between vectors at that position in pos_examples and neg_examples.
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


def dt_accuracy(model, resid_streams):
    # The initial steps for data preparation remain the same
    # Assuming all_indices is available as before
    model.eval()
    all_quantised, all_indices = model.quantized_indices(resid_streams.to(device))
    all_quantised = all_quantised.detach().cpu().numpy()
    all_indices = all_indices.detach().cpu().numpy()
    all_indices = all_indices.T
    
    halfway = all_indices.shape[0] // 2
    labels = np.ones(halfway)
    labels = np.concatenate((labels, np.zeros(halfway)))

    # OneHotEncoder setup and transformation remains the same
    encoder = OneHotEncoder()
    all_indices_encoded = encoder.fit_transform(all_indices).toarray()

    # Feature names for interpretation remains the same
    length = all_indices.shape[1]
    original_feature_names = ["Feature_{}".format(i) for i in range(length)]

    # Mapping each one-hot encoded column back to the original feature and category remains the same
    feature_mapping = []
    for i, categories in enumerate(encoder.categories_):
        feature_mapping.extend([(original_feature_names[i], category) for category in categories])

    # Setup for 5-fold cross-validation
    kf = KFold(n_splits=5)
    accuracies = []

    # Cross-validation loop
    for train_index, test_index in kf.split(all_indices_encoded):
        X_train, X_test = all_indices_encoded[train_index], all_indices_encoded[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train the decision tree model
        tree = DecisionTreeClassifier(max_depth=5)
        tree.fit(X_train, y_train)

        # Get the predictions and calculate accuracy
        predictions = tree.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

    # Calculate mean accuracy across all folds
    mean_accuracy = np.mean(accuracies)

    return mean_accuracy

# Train the model
def train(model, epochs, optimizer, train_streams, eval_streams, train_labels, eval_labels, alpha=0.1, verbose=True, print_epoch=None):

    train_losses, eval_losses, unique_indices_utilised, train_sim_losses, eval_sim_losses, dt_accs = [], [], [], [], [], []

    best_eval_loss = 999
    best_dt_acc = 0.0

    for epoch in tqdm(range(epochs)):
        model.train()  # Set the model to training mode
        epoch_loss = 0

        optimizer.zero_grad()  # Zero the gradients
        outputs, commit_loss = model(train_streams.to(device))  # Forward pass: compute the model output
        loss = loss_function(outputs, train_streams.to(device))  # Compute the loss
        loss += commit_loss[0]
        loss = (1-alpha) * loss

        # Calculate the similarity loss
        positive_indices = torch.where(train_labels == 1.0)[0]
        negative_indices = torch.where(train_labels == 0.0)[0]
        pos_streams = train_streams[:, positive_indices, :]
        neg_streams = train_streams[:, negative_indices, :]

        model.eval()
        pos_quant = model.encode(pos_streams)
        neg_quant = model.encode(neg_streams)
        pos_quant = einops.rearrange(pos_quant, 'S N D -> D S N')
        neg_quant = einops.rearrange(neg_quant, 'S N D -> D S N')
        similarity_loss = alpha * calculate_similarity_loss(pos_quant, neg_quant)
        loss += similarity_loss
        train_sim_losses.append(similarity_loss.item())

        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        epoch_loss += loss.item()  # Accumulate the loss

        divider = epochs // 10 if print_epoch is None else print_epoch

        if epoch % divider == 0:
            # Calculate eval loss
            model.eval()  # Set the model to evaluation mode
            eval_loss = 0
            with torch.no_grad():  # No need to track the gradients
                eval_outputs, eval_commit_loss = model(eval_streams.to(device))  # Forward pass: compute the model output
                eval_loss = loss_function(eval_outputs, eval_streams.to(device))  # Compute the loss
                eval_loss += eval_commit_loss[0]
                eval_loss = (1-alpha) * eval_loss

            # Calculate similarity loss
            with torch.no_grad():
                positive_indices = torch.where(eval_labels == 1.0)[0]
                negative_indices = torch.where(eval_labels == 0.0)[0]
                pos_streams = eval_streams[:, positive_indices, :]
                neg_streams = eval_streams[:, negative_indices, :]
                pos_quant = model.encode(pos_streams)
                neg_quant = model.encode(neg_streams)
                pos_quant = einops.rearrange(pos_quant, 'S N D -> D S N')
                neg_quant = einops.rearrange(neg_quant, 'S N D -> D S N')
                similarity_loss = alpha * calculate_similarity_loss(pos_quant, neg_quant, verbose=True)
                eval_loss += similarity_loss
                eval_sim_losses.append(similarity_loss.item())

            # Calculate decision tree accuracy
            dt_acc = dt_accuracy(model, resid_streams)

            # Update best eval loss
            if config['training']['save_type'] == 'loss':
                if eval_loss.item() < best_eval_loss:
                    print(f"Saving model (eval loss = {eval_loss.item():.4f}, previous best = {best_eval_loss:.4f})")
                    best_eval_loss = eval_loss.item()

                    # Prepare the dictionary to be saved
                    save_dict = {
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'train_losses': train_losses,
                        'eval_losses': eval_losses,
                        'unique_indices': unique_indices_utilised,
                        'train_sim_losses': train_sim_losses,
                        'eval_sim_losses': eval_sim_losses,
                        'dt_accs': dt_accs
                    }

                    # Use the path from your config or another method to determine the save path
                    save_path = config['paths']['model_save'] + f"{args.config.split('.')[0]}" f"_{date_hour_minute_str}.pt"
                    torch.save(save_dict, save_path)

            elif config['training']['save_type'] == 'dt':
                if dt_acc > best_dt_acc:
                    print(f"Saving model (dt acc = {dt_acc:.4f}, previous best = {best_dt_acc:.4f})")
                    best_dt_acc = dt_acc

                    # Prepare the dictionary to be saved
                    save_dict = {
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'train_losses': train_losses,
                        'eval_losses': eval_losses,
                        'unique_indices': unique_indices_utilised,
                        'train_sim_losses': train_sim_losses,
                        'eval_sim_losses': eval_sim_losses,
                        'dt_accs': dt_accs
                    }

                    # Use the path from your config or another method to determine the save path
                    save_path = config['paths']['model_save'] + f"{args.config.split('.')[0]}" f"_{date_hour_minute_str}.pt"
                    torch.save(save_dict, save_path)

            else: print("Wrong save type specified (must be 'dt' or 'loss').")


            # Calculate unique codes on train
            _, train_indices = model.quantized_indices(train_streams.to(device))
            unique_indices = torch.unique(train_indices)
            unique_indices = len(unique_indices)
            unique_indices_utilised.append(unique_indices)

            if verbose:
                print(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss:.3f} (sim loss = {train_sim_losses[-1]:.4f}), Eval Loss: {eval_loss:.3f} (sim loss = {eval_sim_losses[-1]:.4f}), Unique Indices: {unique_indices}, DT Accuracy: {dt_acc:.4f}")
            train_losses.append(epoch_loss)
            eval_losses.append(eval_loss.item())

    return model, train_losses, eval_losses, unique_indices_utilised


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Transformer Autoencoder with a given configuration.")
    parser.add_argument('-c', '--config', type=str, default='configs/default.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration (assumes the argument doesn't have the configs/ prefix)
    config_path = 'configs/' + args.config if 'configs/' not in args.config else args.config
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}.")

    # Extract residual streams from dataset
    resid_streams = torch.load(config['paths']['data'])
    # Rearrange with einops
    resid_streams = einops.rearrange(resid_streams, 'n l d -> l n d')


    # Load configuration
    input_dim = resid_streams.shape[-1]  # Size of the input
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")
    sequence_length = resid_streams.shape[0]  # Length of the sequence
    input_dim = config['model']['input_dim']
    d_model = config['model']['d_model']
    nhead = config['model']['nhead']
    num_encoder_layers = config['model']['num_encoder_layers']
    num_decoder_layers = config['model']['num_decoder_layers']
    dim_feedforward = config['model']['dim_feedforward']
    dropout = config['model']['dropout']


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

    epochs = config['training']['epochs']
    print_epoch = config['training']['print_epoch']
    alpha = config['training']['alpha']
    verbose = config['training']['verbose']
    date_hour_minute_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Instantiate model, optimiser and loss function
    if config['model']['type'] == 'fsq':
        model = TransformerAutoencoder(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=dropout).to(device)
    elif config['model']['type'] == 'vq':
        codebook_size = config['model']['codebook_size']
        codebook_dim = config['model']['codebook_dim']
        threshold_ema_dead_code = config['model']['threshold_ema_dead_code']
        model = TransformerVQVAE(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                       codebook_size=codebook_size, codebook_dim=codebook_dim, threshold_ema_dead_code=threshold_ema_dead_code, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate']) 
    loss_function = nn.MSELoss()

    model, train_losses, eval_losses, unique_indices_ut = train(model, epochs, optimizer, train_streams, eval_streams, train_labels, 
                                                                eval_labels, alpha=alpha, verbose=verbose, print_epoch=print_epoch)