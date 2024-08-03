import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shillml.models import FeedForwardNeuralNet


class ScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, score_act=F.tanh, hutchinson_samples=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hutchinson_samples = hutchinson_samples
        # Two neural networks: one for the score
        score_dims = [input_dim] + hidden_dims + [input_dim]
        score_activations = [score_act] * (len(score_dims) - 2) + [None]
        self.score_net = FeedForwardNeuralNet(score_dims, score_activations)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output tensor.
        """
        score = self.score_net.forward(x)
        # Compute the squared L2 norm of the score
        # score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)
        if self.hutchinson_samples > 0:
            jacobian_trace = self.hutchinson_trace(x, num_samples=self.hutchinson_samples)
        else:
            jacobian = self.score_net.jacobian_network(x)
            jacobian_trace = torch.sum(torch.diagonal(jacobian, dim1=1, dim2=2), dim=1)
        return score, jacobian_trace

    def hutchinson_trace(self, x, num_samples=1):
        """
        Compute the Hutchinson trace estimator.

        :param x: Input tensor.
        :param num_samples: Number of Hutchinson samples.
        :return: Hutchinson trace.
        """
        batch_size, input_dim = x.shape
        v = torch.randn(batch_size, num_samples, input_dim, device=x.device)
        jacobian = self.score_net.jacobian_network(x)
        quadratic_form = torch.einsum('bsi,boi,bsi->bs', v, jacobian, v)
        return quadratic_form.mean(dim=1)


class ScoreBasedMatchingLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(model: ScoreModel, x: Tensor, target=None):
        score, jacobian_trace = model(x)
        # Compute the squared L2 norm of the score
        score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)
        jacobian_trace = torch.mean(jacobian_trace)
        loss = score_norm + 2 * jacobian_trace
        return loss


def mala_sampling(model: ScoreModel, num_samples, step_size, num_steps, burn_in=500, thinning=10, initial_samples=None):
    """
    Generate samples using the Metropolis-Adjusted Langevin Algorithm (MALA).

    :param model: Trained score model.
    :param num_samples: Number of samples to generate after burn-in and thinning.
    :param step_size: Step size for the Langevin Dynamics.
    :param num_steps: Number of MALA steps.
    :param burn_in: Number of burn-in steps.
    :param thinning: Thinning interval to reduce autocorrelation.
    :param initial_samples: Initial samples to start the dynamics.
    :return: Generated samples.
    """
    if initial_samples is None:
        initial_samples = torch.randn((100, model.score_net.output_dim))

    samples = initial_samples
    accepted_samples = []

    for step in range(num_steps):
        current_samples = samples.clone()
        # current_score = model(current_samples)
        current_score = model.score_net.forward(current_samples)
        # Propose new samples
        noise = torch.randn_like(current_samples)
        proposed_samples = current_samples + step_size * current_score + np.sqrt(2 * step_size) * noise
        proposed_score = model.score_net.forward(proposed_samples)

        # Compute acceptance probability
        current_log_prob = -0.5 * torch.sum(current_samples ** 2, dim=1)
        proposed_log_prob = -0.5 * torch.sum(proposed_samples ** 2, dim=1)

        current_to_proposed_log_prob = current_log_prob + torch.sum(
            (proposed_samples - current_samples - step_size * current_score) ** 2, dim=1) / (4 * step_size)
        proposed_to_current_log_prob = proposed_log_prob + torch.sum(
            (current_samples - proposed_samples - step_size * proposed_score) ** 2, dim=1) / (4 * step_size)

        acceptance_prob = torch.exp(proposed_to_current_log_prob - current_to_proposed_log_prob)
        acceptance_prob = torch.min(acceptance_prob, torch.ones_like(acceptance_prob))

        # Accept or reject
        uniform_samples = torch.rand_like(acceptance_prob)
        accepted = (uniform_samples < acceptance_prob).float()
        samples = accepted[:, None] * proposed_samples + (1 - accepted[:, None]) * current_samples

        # Collect samples after burn-in period and according to thinning interval
        if step >= burn_in and (step - burn_in) % thinning == 0:
            accepted_samples.append(samples)

        # Break if we have enough samples
        if len(accepted_samples) >= num_samples:
            break

    return torch.cat(accepted_samples)


def generate_toy_data(num_samples):
    """
    Generate toy data for training.

    :param num_samples: Number of samples to generate.
    :return: Generated samples.
    """
    centers = torch.tensor([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    mixture = torch.randint(0, 4, (num_samples,))
    samples = centers[mixture] + 0.1 * torch.randn(num_samples, 2)
    return samples


if __name__ == "__main__":
    from shillml.utils import fit_model
    input_dim = 2  # Example input dimension
    hidden_dims = [64]  # Example hidden dimensions
    model = ScoreModel(input_dim, hidden_dims)

    data = generate_toy_data(8000)
    score_loss = ScoreBasedMatchingLoss()
    # Need batching
    fit_model(model, score_loss, data, [data], epochs=100, batch_size=128)
    # Initialize with random samples
    samples = mala_sampling(
        model,
        num_samples=1000,
        step_size=0.001,
        num_steps=10000,
        burn_in=500,
        thinning=10
    )
    data = data.detach()
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
    plt.title("Original Data")

    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0].detach(), samples[:, 1].detach(), alpha=0.5, s=1)
    plt.title("Generated Samples")

    plt.tight_layout()
    plt.show()