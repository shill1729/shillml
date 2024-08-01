import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from shillml.ffnn import FeedForwardNeuralNet
from shillml.losses import CovarianceMSELoss


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


class ScoreBasedDiffusion(ScoreModel):
    def __init__(self, input_dim, hidden_dims, score_act=F.tanh, cov_act=F.tanh, hutchinson_samples=0, *args, **kwargs):
        # Neural network for covariance
        super().__init__(input_dim, hidden_dims, score_act, hutchinson_samples, *args, **kwargs)
        cov_dims = [input_dim] + hidden_dims + [input_dim ** 2]
        cov_activations = [cov_act] * (len(cov_dims) - 2) + [None]
        self.covariance_net = FeedForwardNeuralNet(cov_dims, cov_activations)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        score = self.score_net.forward(x)
        # Compute the squared L2 norm of the score
        # score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)
        if self.hutchinson_samples > 0:
            jacobian_trace = self.hutchinson_trace(x, num_samples=self.hutchinson_samples)
        else:
            jacobian = self.score_net.jacobian_network(x)
            jacobian_trace = torch.sum(torch.diagonal(jacobian, dim1=1, dim2=2), dim=1)
        d = self.score_net.output_dim
        cov = self.covariance_net.forward(x).view((x.size(0), d, d))
        cov_div = self.covariance_divergence(x, cov)
        return score, jacobian_trace, cov, cov_div

    @staticmethod
    def covariance_divergence(x: Tensor, cov: Tensor) -> Tensor:
        batch_size, d, _ = cov.size()
        cov_div = torch.zeros(batch_size, d, device=x.device)
        for i in range(d):
            for j in range(d):
                grad_cov = torch.autograd.grad(outputs=cov[:, i, j], inputs=x,
                                               grad_outputs=torch.ones_like(cov[:, i, j]),
                                               retain_graph=True, create_graph=True)[0]
                cov_div[:, i] += grad_cov[:, j]
        return cov_div


class ScoreBasedMatchingLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(model_output: Tensor, targets=None):
        score, jacobian_trace = model_output
        # Compute the squared L2 norm of the score
        score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)
        jacobian_trace = torch.mean(jacobian_trace)
        loss = score_norm + 2 * jacobian_trace
        return loss


class ScoreBasedMatchingDiffusionLoss(nn.Module):
    def __init__(self, covariance_weight=1., stationary_weight=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.covariance_weight = covariance_weight
        self.stationary_weight = stationary_weight
        self.matrix_mse = CovarianceMSELoss()

    def forward(self, model_output: Tensor, targets: Tensor):
        score, jacobian_trace, model_cov, model_cov_div = model_output
        mu, cov = targets
        # Compute the squared L2 norm of the score
        score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)
        jacobian_trace = torch.mean(jacobian_trace)
        score_loss = score_norm + 2 * jacobian_trace
        covariance_loss = self.matrix_mse.forward(model_cov, cov)
        cov_inv = torch.linalg.inv(cov)
        stationary_target = torch.bmm(cov_inv, 2*mu.unsqueeze(2)-model_cov_div).squeeze()
        stationary_loss = torch.linalg.vector_norm(score-stationary_target, ord=2, dim=1)
        stationary_loss = torch.mean(stationary_loss)
        total_loss = score_loss + covariance_loss + stationary_loss
        return total_loss


def train_score_model(model, data, num_epochs, batch_size, lr, hutchinson_samples=0, weight_decay=1e-3):
    """
    Train the score model using the provided data.

    :param model: The ScoreModel to be trained.
    :param data: Training data.
    :param num_epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param lr: Learning rate.
    :param hutchinson_samples: Number of samples for Hutchinson's trace estimator.
    :param weight_decay: Weight decay coefficient.
    """
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            loss = model.score_matching_loss(x, hutchinson_samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


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
    from shillml.losses import fit_model
    input_dim = 2  # Example input dimension
    hidden_dims = [64, 64]  # Example hidden dimensions
    model = ScoreModel(input_dim, hidden_dims)

    data = generate_toy_data(8000)
    score_loss = ScoreBasedMatchingLoss()
    fit_model(model, score_loss, data, data, epochs=100, batch_size=128)
    # Initialize with random samples
    samples = mala_sampling(
        model,
        num_samples=1000,
        step_size=0.01,
        num_steps=10000,
        burn_in=500,
        thinning=10
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
    plt.title("Original Data")

    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0].detach(), samples[:, 1].detach(), alpha=0.5, s=1)
    plt.title("Generated Samples")

    plt.tight_layout()
    plt.show()
