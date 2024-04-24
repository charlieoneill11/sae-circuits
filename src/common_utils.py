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


def feature_string_to_head_and_layer(feature_index, head_labels):

    extraction = head_labels[feature_index]

    if "mlp" in extraction.lower():
        layer = int(extraction.split("_")[0])
        head = 12
        return layer, head

    # Get head and layer e.g. 'L0H1' -> (0, 1)
    # Layer is everything after L and before H
    layer = int(re.findall(r"L(\d+)H", extraction)[0])
    # Head is everything after H
    head = int(re.findall(r"H(\d+)", extraction)[0])

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


def evaluate_circuit(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Calculate ROC AUC
    roc_auc = auc(fpr, tpr)

    # Calculate F1
    f1 = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))

    return fpr, tpr, roc_auc, f1, thresholds


def calculate_f1_score(y_true, y_pred):
    # Calculate True Positives (TP)
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # Calculate False Positives (FP)
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # Calculate False Negatives (FN)
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate True Negatives (TN)
    TN = np.sum((y_true == 0) & (y_pred == 0))

    # Calculate Precision and Recall
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    F1 = (
        2 * (Precision * Recall) / (Precision + Recall)
        if (Precision + Recall) > 0
        else 0
    )

    return F1, Precision, Recall, TP, FP, TN, FN


def head_labels_to_ground_truth(head_labels, ground_truth):
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

    return layers, heads, ground_truth_array
