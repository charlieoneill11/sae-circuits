import torch
import torch.nn.functional as F
import einops
from tqdm import tqdm


def loss_fn(decoded_activations, learned_activations, resid_streams, lambda_=0.01):
    """
    Loss function for the sparse autoencoder. The loss is a combination of the reconstruction loss and the sparsity loss.

    Args:
        decoded_activations: the activations decoded from the learned activations
        learned_activations: the activations learned by the model
        resid_streams: the residual streams
        lambda_: the weight of the sparsity loss

    Returns:
        The total loss
    """

    # RECONSTRUCTION LOSS
    recon_loss = F.mse_loss(decoded_activations, resid_streams)

    # SPARSITY LOSS
    learned_activations_flat = einops.rearrange(learned_activations, "b s n -> (b s) n")
    sparsity_loss = torch.mean(torch.norm(learned_activations_flat, p=1, dim=1))

    # combine
    return recon_loss + (lambda_ * sparsity_loss)


def train(model, n_epochs, optimizer, train_streams, eval_streams, lambda_=0.01):
    """
    Train the sparse autoencoder model.

    Args:
        model: the sparse autoencoder model
        n_epochs: the number of epochs to train for
        optimizer: the optimizer
        train_streams: the training streams
        eval_streams: the evaluation streams
        lambda_: the weight of the sparsity loss

    Returns:
        The trained model
    """
    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        learned_activations, decoded_activations = model(train_streams)
        loss = loss_fn(
            decoded_activations, learned_activations, train_streams, lambda_=lambda_
        )
        loss.backward()
        optimizer.step()
        if epoch % (n_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                eval_learned_activations, eval_decoded_activations = model(eval_streams)
                eval_loss = loss_fn(
                    eval_decoded_activations,
                    eval_learned_activations,
                    eval_streams,
                    lambda_=lambda_,
                )
                print(
                    f"Train loss = {loss.item():.4f}, Eval loss = {eval_loss.item():.4f}"
                )
    return model
