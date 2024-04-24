"""
This script runs an experiment to test the effect of different numbers of examples for training the Sparse Autoencoder (SAE)
on three tasks: 'ioi' (Indirect Object Identification), 'gt' (Greater-than), and 'ds' (Docstring).
The numbers of SAE training examples tested are [5, 10, 25, 50, 100, 250, and 500].
The script evaluates the model's performance using the ROC AUC metric and reports the mean and standard deviation
of the scores across multiple trials for each combination of task and number of SAE training examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append("../src")

from sae import SparseAutoencoder
from train import train
from common_utils import head_labels_to_ground_truth
from main import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    evaluate_node_circuit,
    evaluate_edge_circuit,
)


# Default configuration values
task_mappings = {
    "gt": "Greater-than",
    "ioi": "Indirect Object Identification",
    "ds": "Docstring",
}
num_unique = 300
n_epochs = 500
num_trials = 3
lambda_ = 0.02
normalise = False
learning_rate = 0.001
data_dir = "../data"
model_dir = "../models"

results = {}


def load_data(task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir_path = Path(data_dir)
    resid_streams = torch.load(data_dir_path / f"{task}/resid_streams.pt").to(device)
    head_labels = torch.load(data_dir_path / f"{task}/labels.pt")
    ground_truth = torch.load(data_dir_path / f"{task}/ground_truth.pt")
    return resid_streams, head_labels, ground_truth


def prepare_data(resid_streams, num_sae_examples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = torch.ones(resid_streams.shape[0] // 2)
    labels = torch.cat((labels, torch.zeros_like(labels)))
    permutation = torch.randperm(resid_streams.shape[0])
    resid_shuffled = resid_streams[permutation, :, :]
    train_streams = resid_shuffled[:num_sae_examples, :, :].to(device)
    eval_streams = resid_shuffled[num_sae_examples:, :, :].to(device)
    return train_streams, eval_streams


def train_model(model, optimizer, train_streams, eval_streams):
    model = train(
        model, n_epochs, optimizer, train_streams, eval_streams, lambda_=lambda_
    )
    model = model.to("cpu")
    return model


def evaluate_model(model, resid_streams, head_labels, ground_truth, task_type):
    _, heads, ground_truth_array = head_labels_to_ground_truth(
        head_labels, ground_truth
    )
    if task_type == "node":
        roc_auc = evaluate_node_circuit(
            model, resid_streams, head_labels, ground_truth_array, normalise
        )
    elif task_type == "edge":
        roc_auc = evaluate_edge_circuit(
            model, resid_streams, head_labels, ground_truth_array, normalise, len(heads)
        )
    else:
        raise ValueError("Task type must be either 'node' or 'edge'")
    return roc_auc


def main():
    tasks = ["ioi", "gt", "ds"]
    num_sae_examples_list = [5, 10, 25, 50, 100, 250, 500]

    for task in tasks:
        print(f"Task: {task_mappings[task]}")

        for num_sae_examples in num_sae_examples_list:
            print(f"Number of SAE training examples: {num_sae_examples}")

            roc_results = []

            for i in range(num_trials):
                resid_streams, head_labels, ground_truth = load_data(task)
                train_streams, eval_streams = prepare_data(
                    resid_streams, num_sae_examples
                )

                model = SparseAutoencoder(
                    n_input_features=resid_streams.shape[-1],
                    n_learned_features=num_unique,
                    geometric_median_dataset=None,
                ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                model = train_model(model, optimizer, train_streams, eval_streams)

                resid_streams = resid_streams.to("cpu")
                roc_auc = evaluate_model(
                    model, resid_streams, head_labels, ground_truth, task_type="node"
                )
                roc_results.append(roc_auc)

            print(
                f"Mean ROC = {np.mean(roc_results):.4f} (+/- {np.std(roc_results):.4f})\n"
            )

            if task not in results:
                results[task] = {}
            results[task][num_sae_examples] = {
                "mean_roc": np.mean(roc_results),
                "std_roc": np.std(roc_results),
            }


def save_results(results):
    script_name = Path(__file__).stem
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{script_name}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
    save_results(results)
