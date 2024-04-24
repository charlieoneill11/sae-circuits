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

from common_utils import gen_array_template, feature_string_to_head_and_layer, softmax


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
                    normalized_matrix[h1, h2, :, :] = (
                        co_occurrence_matrix[h1, h2, :, :] / total_co_occurrences
                    )

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
                    total_co_occurrences = np.sum(
                        positive_matrix[h1, h2, :, :] > 0
                    ) + np.sum(negative_matrix[h1, h2, :, :] > 0)
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


def softmax_edge(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def positive_and_negative_cooccurrence(codes, learned_activations):
    halfway = codes.shape[0] // 2
    positive_indices = codes[:halfway, :]
    negative_indices = codes[halfway:, :]

    # Assume all_indices, positive_indices, and negative_indices are defined, as well as n_heads and n_feat
    n_feat = learned_activations.shape[-1]
    n_heads = codes.shape[1]
    positive_co_occurrence_matrix = gen_co_occurrence_matrix(
        positive_indices, n_heads, n_feat
    )
    negative_co_occurrence_matrix = gen_co_occurrence_matrix(
        negative_indices, n_heads, n_feat
    )

    return positive_co_occurrence_matrix, negative_co_occurrence_matrix


def edge_circuit_prediction(
    positive_cooccurrence_matrix,
    negative_cooccurrence_matrix,
    head_labels,
    ground_truth_array,
    normalise,
    k,
):
    unique_co_occurrence_counts = unique_co_occurrences(
        positive_cooccurrence_matrix, negative_cooccurrence_matrix, normalise=normalise
    )

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

    y_pred = np.zeros_like(ground_truth_array)
    for l, h in circuit_components[:k]:
        y_pred[l, h] += 1

    # Flatten the arrays
    y_true = ground_truth_array.flatten()
    y_pred = y_pred.flatten()

    # Normalise y_pred with softmax
    y_pred = softmax_edge(y_pred)

    return y_pred, y_true
