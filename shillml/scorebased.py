import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from shillml.ffnn import FeedForwardNeuralNet


class EnhancedScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation=F.tanh):
        """
        Initialize the EnhancedScoreModel with given dimensions and activation function.

        :param input_dim: Dimension of the input features.
        :param hidden_dims: List of dimensions for hidden layers.
        :param activation: Activation function to be used in the hidden layers.
        """
        super(EnhancedScoreModel, self).__init__()
        dims = [input_dim] + hidden_dims + [input_dim]
        dims2 = [input_dim] + hidden_dims + [input_dim*input_dim]
        activations = [activation] * (len(dims) - 2) + [None]

        # Neural network for mu
        self.mu_net = FeedForwardNeuralNet(dims, activations)

        # Neural network for sigma (matrix)
        self.sigma_net = FeedForwardNeuralNet(dims2, activations)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Computed score.
        """
        mu = self.mu_net(x)
        sigma = self.sigma_net(x)

        # Reshape sigma to be a square matrix
        sigma = sigma.view(x.shape[0], x.shape[1], x.shape[1])

        # Compute Sigma = sigma * sigma^T
        Sigma = torch.bmm(sigma, sigma.transpose(1, 2))

        # Compute Sigma^-1
        Sigma_inv = torch.inverse(Sigma)

        # Compute divergence of Sigma
        div_Sigma = self.compute_divergence(Sigma)

        # Compute the score: Sigma^-1 * (2*mu - div_Sigma)
        score = torch.bmm(Sigma_inv, (2 * mu - div_Sigma).unsqueeze(-1)).squeeze(-1)

        return score

    def compute_divergence(self, Sigma):
        """
        Compute the divergence of Sigma.

        :param Sigma: Input tensor of shape (batch_size, input_dim, input_dim).
        :return: Divergence of Sigma.
        """
        batch_size, input_dim, _ = Sigma.shape
        div_Sigma = torch.zeros(batch_size, input_dim, device=Sigma.device)

        params = list(self.sigma_net.parameters())

        for i in range(input_dim):
            grads = torch.autograd.grad(Sigma[:, i, :].sum(), params, create_graph=True)
            div_Sigma[:, i] = sum(grad.sum(dim=tuple(range(1, grad.dim()))) for grad in grads)

        return div_Sigma

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
            jacobian = self.compute_jacobian(x)
            jacobian_trace = torch.sum(torch.diagonal(jacobian, dim1=1, dim2=2), dim=1)
        jacobian_trace = torch.mean(jacobian_trace)

        loss = score_norm + 2 * jacobian_trace
        return loss

    def compute_jacobian(self, x):
        """
        Compute the Jacobian of the score function.

        :param x: Input tensor.
        :return: Jacobian of the score function.
        """
        batch_size, input_dim = x.shape
        jacobian = torch.zeros(batch_size, input_dim, input_dim, device=x.device)

        for i in range(input_dim):
            jacobian[:, i, :] = torch.autograd.grad(self.forward(x)[:, i].sum(), x, create_graph=True)[0]

        return jacobian

    def hutchinson_trace(self, x, num_samples=1):
        """
        Compute the Hutchinson trace estimator.

        :param x: Input tensor.
        :param num_samples: Number of Hutchinson samples.
        :return: Hutchinson trace.
        """
        batch_size, input_dim = x.shape
        v = torch.randn(batch_size, num_samples, input_dim, device=x.device)
        jacobian = self.compute_jacobian(x)
        quadratic_form = torch.einsum('bsi,boi,bsi->bs', v, jacobian, v)
        return quadratic_form.mean(dim=1)


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
