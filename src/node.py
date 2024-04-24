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

from common_utils import (
    gen_array_template,
    feature_string_to_head_and_layer,
    softmax,
    evaluate_circuit,
)


def node_circuit_prediction(
    codes, ground_truth_array, head_labels, normalise=False, across_layer=False
):

    # Find number of unique positive and unique negative codes per head
    unique_to_positive_array, unique_to_negative_array = (
        unique_to_positives_and_negatives(codes, head_labels, normalise=normalise)
    )

    # If across_layer, we apply softmax to the rows (layers) of the array
    if across_layer:
        unique_to_positive_array = softmax(unique_to_positive_array, axis=0)
        unique_to_negative_array = softmax(unique_to_negative_array, axis=0)

    # Flatten the arrays
    y_true = np.array(ground_truth_array).flatten()
    y_pred = unique_to_positive_array.flatten()

    # Normalise y_pred with softmax
    if not across_layer:
        y_pred = softmax(y_pred, axis=0)

    return y_pred, y_true


def unique_to_positives_and_negatives(codes, head_labels, normalise=False):

    # Negative and positive indices
    halfway = codes.shape[0] // 2
    positive_codes = codes[:halfway, :]
    negative_codes = codes[halfway:, :]

    unique_to_positive_array = gen_array_template(head_labels)
    unique_to_negative_array = gen_array_template(head_labels)

    for i in range(len(head_labels)):
        # Calculate head and layer
        layer, head = feature_string_to_head_and_layer(i, head_labels)

        positive = set(positive_codes[:, i].tolist())
        negative = set(negative_codes[:, i].tolist())
        total_unique = positive.union(negative)

        # In positive but not negative
        unique_to_positive = list(positive - negative)
        # In negative but not positive
        unique_to_negative = list(negative - positive)

        if normalise:
            # Normalise by total number of unique indices
            unique_to_positive_array[layer, head] = len(unique_to_positive) / len(
                total_unique
            )
            unique_to_negative_array[layer, head] = len(unique_to_negative) / len(
                total_unique
            )

        else:
            # Set the values
            unique_to_positive_array[layer, head] = len(unique_to_positive)
            unique_to_negative_array[layer, head] = len(unique_to_negative)

    return unique_to_positive_array, unique_to_negative_array
