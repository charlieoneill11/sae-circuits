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
from common_utils import *



### MAIN CODE ###

task = 'gt'
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
num_trials = 5
models = ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

overall_results = {}

for model_name in models:
    roc_results = []
    # Load residual streams
    resid_streams = torch.load(f"../data/{task}/model_sizes/resid_heads_mean_{model_name}.pt").to(device)
    # Resid streams are shape n_examples x n_heads x d_model
    # Expand them 10 times along n_examples by repeating along this dimension
    resid_streams = einops.repeat(resid_streams, 'n_h d -> (10 n_h) d')


    head_labels = torch.load(f'../data/{task}/model_sizes/labels_heads_mean.pt')
    ground_truth = torch.load(f'../data/{task}/model_sizes/ground_truth.pt')

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

    for j in range(num_trials):
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
        cutoff = int(resid_shuffled.shape[0] * 0.8)
        train_streams = resid_shuffled[:cutoff, :, :].to(device)
        train_labels = labels_shuffled[:cutoff].to(device)
        eval_streams = resid_shuffled[cutoff:, :, :].to(device)
        eval_labels = labels_shuffled[cutoff:].to(device)


        model = SparseAutoencoder(n_input_features=resid_streams.shape[-1], n_learned_features=num_unique, geometric_median_dataset=None).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model = train(model, n_epochs, optimizer, train_streams, eval_streams, lambda_=0.02, alpha_=0.0)
        model = model.to('cpu')
        resid_streams = resid_streams.to('cpu')

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

    overall_results[model_name] = roc_results

print(overall_results)
# Save as JSON to output
import json

with open(f"../output/data/{task}/roc_results_model_sizes.json", 'w') as f:
    json.dump(overall_results, f)