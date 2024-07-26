import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from shillml.ffnn import FeedForwardNeuralNet


class ScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation=F.tanh):
        """
        Initialize the ScoreModel with given dimensions and activation function.

        :param input_dim: Dimension of the input features.
        :param hidden_dims: List of dimensions for hidden layers.
        :param activation: Activation function to be used in the hidden layers.
        """
        super(ScoreModel, self).__init__()
        dims = [input_dim] + hidden_dims + [input_dim]
        activations = [activation] * (len(dims) - 2) + [None]
        self.net = FeedForwardNeuralNet(dims, activations)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.net(x)

    def score_matching_loss(self, x, hutchinson_samples=0):
        """
        Compute the score matching loss.

        :param x: Input tensor.
        :param hutchinson_samples: Number of samples for Hutchinson's trace estimator.
        :return: Computed loss.
        """
        score = self.forward(x)

        # Compute the squared L2 norm of the score
        score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)

        if hutchinson_samples > 0:
            jacobian_trace = self.hutchinson_trace(x, num_samples=hutchinson_samples)
        else:
            jacobian = self.net.jacobian_network(x)
            jacobian_trace = torch.sum(torch.diagonal(jacobian, dim1=1, dim2=2), dim=1)
        jacobian_trace = torch.mean(jacobian_trace)

        loss = score_norm + 2 * jacobian_trace
        return loss

    def hutchinson_trace(self, x, num_samples=1):
        """
        Compute the Hutchinson trace estimator.

        :param x: Input tensor.
        :param num_samples: Number of Hutchinson samples.
        :return: Hutchinson trace.
        """
        batch_size, input_dim = x.shape
        v = torch.randn(batch_size, num_samples, input_dim, device=x.device)
        jacobian = self.net.jacobian_network(x)
        quadratic_form = torch.einsum('bsi,boi,bsi->bs', v, jacobian, v)
        return quadratic_form.mean(dim=1)


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


def sample_langevin_dynamics(model, num_samples, num_steps, step_size, initial_noise_std):
    """
    Generate samples using Langevin dynamics.

    :param model: Trained ScoreModel.
    :param num_samples: Number of samples to generate.
    :param num_steps: Number of Langevin steps.
    :param step_size: Step size for Langevin dynamics.
    :param initial_noise_std: Standard deviation of the initial noise.
    :return: Generated samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        x = torch.randn(num_samples, model.net.input_dim, device=device) * initial_noise_std
        for _ in range(num_steps):
            score = model(x)
            noise = torch.randn_like(x)
            x = x + 0.5 * step_size * score + torch.sqrt(torch.tensor(step_size, device=device)) * noise
        return x


def sample_mcmc_langevin(model, num_samples, num_steps, step_size, initial_noise_std, temperature=1.0, thinning=100):
    """
    Generate samples using MCMC Langevin dynamics.

    :param model: Trained ScoreModel.
    :param num_samples: Number of samples to generate.
    :param num_steps: Number of Langevin steps.
    :param step_size: Step size for Langevin dynamics.
    :param initial_noise_std: Standard deviation of the initial noise.
    :param temperature: Temperature parameter for MCMC.
    :param thinning: Thinning interval for sampling.
    :return: Generated samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randn(num_samples, model.net.input_dim, device=device) * initial_noise_std

    accepted_samples = torch.zeros((num_samples * (num_steps // thinning), model.net.input_dim), device=device)
    acceptance_count = 0
    sample_index = 0

    for step in range(num_steps):
        with torch.no_grad():
            score = model(x)

            noise = torch.randn_like(x)
            x_proposed = x + 0.5 * step_size * score + torch.sqrt(
                torch.tensor(step_size * temperature, device=device)) * noise
            score_proposed = model(x_proposed)

            current_log_prob = -0.5 * torch.sum(x ** 2, dim=1) + torch.sum(score * x, dim=1)
            proposed_log_prob = -0.5 * torch.sum(x_proposed ** 2, dim=1) + torch.sum(score_proposed * x_proposed, dim=1)

            log_accept_prob = (proposed_log_prob - current_log_prob) / temperature
            accept_prob = torch.exp(torch.clamp(log_accept_prob, max=0))

            u = torch.rand(num_samples, device=device)
            mask = u < accept_prob
            x = torch.where(mask.unsqueeze(1), x_proposed, x)

            acceptance_count += mask.sum().item()

            if (step + 1) % thinning == 0:
                accepted_samples[sample_index:sample_index + num_samples] = x
                sample_index += num_samples
    acceptance_rate = acceptance_count / (num_steps * num_samples)
    print(f"Average acceptance rate: {acceptance_rate:.4f}")
    return accepted_samples


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
    input_dim = 2
    hidden_dims = [32]
    model = ScoreModel(input_dim, hidden_dims)

    data = generate_toy_data(8000)

    train_score_model(model, data, num_epochs=500, batch_size=128, lr=1e-3, hutchinson_samples=0)

    samples = sample_mcmc_langevin(model, num_samples=1000, num_steps=5000, step_size=1e-5, initial_noise_std=0.1,
                                   temperature=0.5)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
    plt.title("Original Data")

    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0].detach(), samples[:, 1].detach(), alpha=0.5, s=1)
    plt.title("Generated Samples")

    plt.tight_layout()
    plt.show()
