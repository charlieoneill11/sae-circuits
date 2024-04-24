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

from transformer_lens import HookedTransformer, HookedTransformerConfig

import transformer_lens.utils as utils
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()

### FUNCTION DEFINITIONS ###

# Loss function is MSE (reconstruction loss) + L1 norm of the learned activations + similarity loss
def loss_fn(decoded_activations, learned_activations, resid_streams, lambda_=0.01):

    # RECONSTRUCTION LOSS
    recon_loss = F.mse_loss(decoded_activations, resid_streams)

    # SPARSITY LOSS
    learned_activations_flat = einops.rearrange(learned_activations, 'b s n -> (b s) n')
    sparsity_loss = torch.mean(torch.norm(learned_activations_flat, p=1, dim=1))

    # combine
    return recon_loss + (lambda_ * sparsity_loss)


def train(model, n_epochs, optimizer, train_streams, eval_streams, lambda_=0.01):
    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        learned_activations, decoded_activations = model(train_streams)
        loss = loss_fn(decoded_activations, learned_activations, train_streams, lambda_=lambda_)
        loss.backward()
        optimizer.step()
        if epoch % (n_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                eval_learned_activations, eval_decoded_activations = model(eval_streams)
                eval_loss = loss_fn(eval_decoded_activations, eval_learned_activations,
                                                             eval_streams, lambda_=lambda_)
                print(f"Train loss = {loss.item():.4f}, Eval loss = {eval_loss.item():.4f}")
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

def softmax(x, axis):
    """Return the softmax of x (if x is a vector) or the softmax of each row (if x is a matrix)"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def gen_softmaxed_unique_to_pos(all_indices, ground_truth_array, head_labels, normalise=False, across_layer=False, return_original=False):
    # Negative and positive indices
    halfway = all_indices.shape[0] // 2
    positive_indices = all_indices[:halfway, :]
    negative_indices = all_indices[halfway:, :]

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

    # If across_layer, we apply softmax to the rows (layers) of the array
    if across_layer:
        unique_to_positive_array = softmax(unique_to_positive_array, axis=0)
        unique_to_negative_array = softmax(unique_to_negative_array, axis=0)

    # Plot the ROC curve in plotly
    y_true = ground_truth_array.flatten()
    y_pred = unique_to_positive_array.flatten()

    # Normalise y_pred with softmax
    if not across_layer:
        y_pred = softmax(y_pred, axis=0)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Calculate ROC AUC
    roc_auc = auc(fpr, tpr)

    # Calculate F1
    f1 = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))

    if return_original:
        return y_true, y_pred, fpr, tpr, roc_auc, f1, thresholds, unique_to_positive_array
    else:
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

# Repeat with num edges
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