import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from pathlib import Path

from sae import SparseAutoencoder
from node import unique_to_positives_and_negatives, node_circuit_prediction
from edge import positive_and_negative_cooccurrence, edge_circuit_prediction
from train import train
from common_utils import evaluate_circuit, head_labels_to_ground_truth


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data_dir"])
    resid_streams = torch.load(
        data_dir / config["resid_streams_path"].format(task=config["task"])
    ).to(device)
    head_labels = torch.load(
        data_dir / config["labels_path"].format(task=config["task"])
    )
    ground_truth = torch.load(
        data_dir / config["ground_truth_path"].format(task=config["task"])
    )
    return resid_streams, head_labels, ground_truth


def prepare_data(resid_streams, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = torch.ones(
        resid_streams.shape[0] // 2
    )  # BIG ASSUMPTION: assumes first half is positive and second half is negative
    labels = torch.cat((labels, torch.zeros_like(labels)))
    permutation = torch.randperm(resid_streams.shape[0])
    resid_shuffled = resid_streams[permutation, :, :]
    cutoff = config["num_sae_examples"]
    train_streams = resid_shuffled[:cutoff, :, :].to(device)
    eval_streams = resid_shuffled[cutoff:, :, :].to(device)
    return train_streams, eval_streams


def train_model(model, optimizer, train_streams, eval_streams, config):
    model = train(
        model,
        config["n_epochs"],
        optimizer,
        train_streams,
        eval_streams,
        lambda_=config["lambda_"],
    )
    model = model.to("cpu")
    return model


def save_model(model, config):
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        model,
        model_dir
        / config["model_save_path"].format(
            task=config["task"], task_type=config["task_type"]
        ),
    )


def evaluate_model(model, resid_streams, head_labels, ground_truth, config):
    layers, heads, ground_truth_array = head_labels_to_ground_truth(
        head_labels, ground_truth
    )
    if config["task_type"] == "node":
        roc_auc = evaluate_node_circuit(
            model, resid_streams, head_labels, ground_truth_array, config["normalise"]
        )
    elif config["task_type"] == "edge":
        roc_auc = evaluate_edge_circuit(
            model,
            resid_streams,
            head_labels,
            ground_truth_array,
            config["normalise"],
            len(heads),
        )
    else:
        raise ValueError("Task type must be either 'node' or 'edge'")
    return roc_auc


def main(config):
    print(f"Type: {config['task_type']}")
    print(f"Task: {config['task_mappings'][config['task']]}")
    print(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    roc_results = []

    for i in range(config["num_trials"]):
        resid_streams, head_labels, ground_truth = load_data(config)
        train_streams, eval_streams = prepare_data(resid_streams, config)

        model = SparseAutoencoder(
            n_input_features=resid_streams.shape[-1],
            n_learned_features=config["num_unique"],
            geometric_median_dataset=None,
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        model = train_model(model, optimizer, train_streams, eval_streams, config)
        save_model(model, config)

        resid_streams = resid_streams.to("cpu")
        roc_auc = evaluate_model(
            model, resid_streams, head_labels, ground_truth, config
        )
        roc_results.append(roc_auc)

    print(roc_results)
    print(f"Mean ROC = {np.mean(roc_results):.4f} (+/- {np.std(roc_results):.4f})")


def evaluate_node_circuit(
    model, resid_streams, head_labels, ground_truth_array, normalise
):
    model.eval()
    learned_activations = model(resid_streams)[0].detach().cpu().numpy()
    codes = np.argmax(learned_activations, axis=2)

    # Circuit prediction and ground truth
    y_pred, y_true = node_circuit_prediction(
        codes, ground_truth_array, head_labels, normalise=normalise, across_layer=False
    )

    # Evaluate circuit
    fpr, tpr, roc_auc, f1, thresholds = evaluate_circuit(y_true, y_pred)

    # Print best f1 score (and corresponding threshold)
    node_best_f1 = np.max(f1)
    best_threshold = thresholds[np.argmax(f1)]
    print(f"Best F1 score: {node_best_f1:.4f}")

    # Print ROC AUC
    print(f"ROC AUC: {roc_auc:.4f}\n\n")

    return roc_auc


def evaluate_edge_circuit(
    model, resid_streams, head_labels, ground_truth_array, normalise, num_heads
):
    # Learned activations and then take argmax to discretise to get codes
    learned_activations = model(resid_streams)[0].detach().cpu().numpy()
    codes = np.argmax(learned_activations, axis=2)

    # Set k to be half of n_heads ** 2
    k = (num_heads**2) // 2

    # Get the edge circuit prediction
    positive_cooccurrence_matrix, negative_cooccurrence_matrix = (
        positive_and_negative_cooccurrence(codes, learned_activations)
    )
    y_pred, y_true = edge_circuit_prediction(
        positive_cooccurrence_matrix,
        negative_cooccurrence_matrix,
        head_labels,
        ground_truth_array,
        normalise=normalise,
        k=k,
    )

    # Evaluate circuit
    fpr, tpr, roc_auc, f1, thresholds = evaluate_circuit(y_true, y_pred)

    # Print best f1 score (and corresponding threshold)
    edge_best_f1 = np.max(f1)
    best_threshold = thresholds[np.argmax(f1)]
    print(f"Best F1 score: {edge_best_f1:.4f}")

    # Print ROC AUC
    print(f"ROC AUC: {roc_auc:.4f}\n\n")

    return roc_auc


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)
