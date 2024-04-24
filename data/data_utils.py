import io
from logging import warning
from typing import Union, List
from site import PREFIXES
import warnings
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import random as rd
import copy
import random
from typing import List, Union
from pathlib import Path
import torch
from transformer_lens import HookedTransformer


# Set up the model
def prompt_to_resid_stream(
    prompt: str,
    model: HookedTransformer,
    resid_type: str = "accumulated",
    position: str = "last",
) -> torch.Tensor:
    """
    Convert a prompt to a residual stream of size (n_layers, d_model)
    """
    # Run the model over the prompt
    with torch.no_grad():
        # If prompt is single dim, unsqueeze it
        _, cache = model.run_with_cache(prompt)

        # Get the accumulated residuals
        if resid_type == "accumulated":
            resid, _ = cache.accumulated_resid(return_labels=True, apply_ln=True)
        elif resid_type == "decomposed":
            resid, _ = cache.decompose_resid(return_labels=True)
        elif resid_type == "heads":
            cache.compute_head_results()
            head_resid, head_labels = cache.stack_head_results(return_labels=True)
            resid = head_resid
            labels = head_labels
        else:
            raise ValueError(
                "resid_type must be one of 'accumulated', 'decomposed', 'heads'"
            )

    # POSITION
    if position == "last":
        last_token_accum = resid[:, 0, -1, :]  # layer, batch, pos, d_model
    elif position == "mean":
        last_token_accum = resid.mean(dim=2).squeeze()
    else:
        raise ValueError("position must be one of 'last', 'mean'")
    return last_token_accum, labels


def all_prompts_to_resid_streams_gt(
    prompts, prompts_cf, model, resid_type="accumulated", position="mean"
):
    """
    Convert all prompts and counterfactual prompts to residual streams
    """
    # Stack prompts and prompts cf
    resid_streams = []
    # Combine the lists of strs
    all_prompts = prompts + prompts_cf
    for i in tqdm(range(len(all_prompts))):
        prompt = all_prompts[i]
        resid_stream, labels = prompt_to_resid_stream(
            prompt, model, resid_type, position
        )
        resid_streams.append(resid_stream)
    # Stack the residual streams into a single tensor
    return torch.stack(resid_streams), labels


def all_prompts_to_resid_streams_ioi(
    prompts, prompts_cf, model, tokenizer, resid_type="accumulated", position="mean"
):
    """
    Convert all prompts and counterfactual prompts to residual streams
    """
    # Stack prompts and prompts cf
    resid_streams, final_prompts = [], []
    all_prompts = torch.cat([prompts, prompts_cf], dim=0)
    for i in tqdm(range(all_prompts.shape[0])):
        prompt = tokenizer.decode(all_prompts[i])
        # Strip the prompt of any exclamation marks
        prompt = prompt.replace("!", "")
        resid_stream, labels = prompt_to_resid_stream(
            prompt, model, resid_type, position
        )
        resid_streams.append(resid_stream)
    # Stack the residual streams into a single tensor
    return torch.stack(resid_streams), labels
