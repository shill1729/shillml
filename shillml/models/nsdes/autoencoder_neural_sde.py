import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Callable


# Assuming FeedForwardNeuralNet is defined similarly to this:
class FeedForwardNeuralNet(nn.Module):
    def __init__(self, neurons: List[int], activations: List[Callable]):
        super().__init__()
        layers = []
        for in_features, out_features, activation in zip(neurons[:-1], neurons[1:], activations):
            layers.append(nn.Linear(in_features, out_features))
            if activation is not None:
                layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    # Placeholder methods for Jacobian and Hessian (implement as needed)
    def jacobian_network(self, x):
        pass

    def hessian_network(self, x):
        pass


# AutoEncoder class using your existing code structure
class AutoEncoder(nn.Module):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_activation: Callable,
                 decoder_activation: Callable):
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.extrinsic_dim = extrinsic_dim

        # Encoder
        encoder_neurons = [extrinsic_dim] + hidden_dims + [intrinsic_dim]
        encoder_activations = [encoder_activation for _ in range(len(encoder_neurons) - 2)] + [None]
        self.encoder = FeedForwardNeuralNet(encoder_neurons, encoder_activations)

        # Decoder
        decoder_neurons = [intrinsic_dim] + hidden_dims[::-1] + [extrinsic_dim]
        decoder_activations = [decoder_activation for _ in range(len(decoder_neurons) - 2)] + [None]
        self.decoder = FeedForwardNeuralNet(decoder_neurons, decoder_activations)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# Neural SDE class adapted to work in the latent space
class LatentNeuralSDE(nn.Module):
    def __init__(self, intrinsic_dim: int, hidden_dim: List[int], drift_activation: Callable,
                 diffusion_activation: Callable, noise_dim: int):
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.noise_dim = noise_dim

        # Drift network
        neurons_mu = [intrinsic_dim] + hidden_dim + [intrinsic_dim]
        activations_mu = [drift_activation for _ in range(len(neurons_mu) - 2)] + [None]
        self.drift_net = FeedForwardNeuralNet(neurons_mu, activations_mu)

        # Diffusion network
        neurons_sigma = [intrinsic_dim] + hidden_dim + [intrinsic_dim * noise_dim]
        activations_sigma = [diffusion_activation for _ in range(len(neurons_sigma) - 2)] + [nn.Softplus()]
        self.diffusion_net = FeedForwardNeuralNet(neurons_sigma, activations_sigma)

    def forward(self, z):
        """
        z: shape (N, n, intrinsic_dim)
        """
        z1 = z[:, :-1, :]  # shape (N, n-1, intrinsic_dim)
        N, n_minus_1, intrinsic_dim = z1.shape
        mu = self.drift_net(z1)  # shape (N, n-1, intrinsic_dim)
        sigma = self.diffusion_net(z1)  # shape (N, n-1, intrinsic_dim * noise_dim)
        sigma = sigma.view(N, n_minus_1, intrinsic_dim, self.noise_dim)
        return mu, sigma


# Functions to compute NLL in the latent space
def log_det_cov(cov):
    """
    cov: shape (N, n, intrinsic_dim, intrinsic_dim)
    """
    # Add small value to diagonal for numerical stability
    epsilon = 1e-6
    cov = cov + epsilon * torch.eye(cov.shape[-1]).to(cov.device)
    log_det = torch.logdet(cov)
    # Replace NaNs and Infs resulting from logdet computation
    log_det = torch.where(torch.isfinite(log_det), log_det, torch.tensor(0.0).to(cov.device))
    return log_det  # shape (N, n)


def mahalanobis_distance(z, mu, cov):
    """
    z: shape (N, n, intrinsic_dim)
    mu: shape (N, n, intrinsic_dim)
    cov: shape (N, n, intrinsic_dim, intrinsic_dim)
    """
    epsilon = 1e-6
    cov_inv = torch.inverse(cov + epsilon * torch.eye(cov.shape[-1]).to(cov.device))
    delta = z - mu
    delta = delta.unsqueeze(-1)  # shape (N, n, intrinsic_dim, 1)
    m_dist = torch.matmul(torch.matmul(delta.transpose(-2, -1), cov_inv), delta)
    m_dist = m_dist.squeeze(-1).squeeze(-1)  # shape (N, n)
    return m_dist


def gaussian_nll(z, mu, cov):
    """
    z: shape (N, n, intrinsic_dim)
    mu: shape (N, n, intrinsic_dim)
    cov: shape (N, n, intrinsic_dim, intrinsic_dim)
    """
    log_det_term = log_det_cov(cov)  # shape (N, n)
    m_dist = mahalanobis_distance(z, mu, cov)  # shape (N, n)
    nll = 0.5 * (m_dist + log_det_term + cov.shape[-1] * torch.log(torch.tensor(2 * torch.pi)))
    return nll.mean()


def euler_maruyama_nll(z, mu, sigma, h):
    """
    z: shape (N, n, intrinsic_dim)
    mu: shape (N, n-1, intrinsic_dim)
    sigma: shape (N, n-1, intrinsic_dim, noise_dim)
    h: float
    """
    z1 = z[:, :-1, :]  # shape (N, n-1, intrinsic_dim)
    z2 = z[:, 1:, :]  # shape (N, n-1, intrinsic_dim)
    delta_z = z2 - z1  # shape (N, n-1, intrinsic_dim)
    drift = mu * h  # shape (N, n-1, intrinsic_dim)
    # Compute covariance matrix
    sigma_t = sigma.transpose(-1, -2)  # shape (N, n-1, noise_dim, intrinsic_dim)
    cov = torch.matmul(sigma, sigma_t) * h  # shape (N, n-1, intrinsic_dim, intrinsic_dim)
    nll = gaussian_nll(delta_z, drift, cov)
    return nll


# Combined model
class AutoencoderNeuralSDE(nn.Module):
    def __init__(self, extrinsic_dim: int, intrinsic_dim: int, autoencoder_hidden_dims: List[int],
                 sde_hidden_dims: List[int], encoder_activation: Callable, decoder_activation: Callable,
                 drift_activation: Callable, diffusion_activation: Callable, noise_dim: int):
        super().__init__()
        self.autoencoder = AutoEncoder(extrinsic_dim, intrinsic_dim, autoencoder_hidden_dims,
                                       encoder_activation, decoder_activation)
        self.neural_sde = LatentNeuralSDE(intrinsic_dim, sde_hidden_dims, drift_activation, diffusion_activation,
                                          noise_dim)

    def forward(self, x):
        x_recon, z = self.autoencoder(x)
        mu, sigma = self.neural_sde(z)
        return x_recon, z, mu, sigma

    def compute_loss(self, x, h, lambda_rec):
        """
        x: shape (N, n, extrinsic_dim)
        h: time step size
        lambda_rec: weighting parameter for reconstruction loss
        """
        x_recon, z, mu, sigma = self.forward(x)
        rec_loss = nn.MSELoss()(x_recon, x)
        nll_loss = euler_maruyama_nll(z, mu, sigma, h)
        total_loss = nll_loss + lambda_rec * rec_loss
        return total_loss, rec_loss, nll_loss


# Training function
def train(model, data_loader, num_epochs, h, lambda_rec, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        total_loss_epoch = 0.0
        rec_loss_epoch = 0.0
        nll_loss_epoch = 0.0
        for batch_idx, (x_batch,) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            total_loss, rec_loss, nll_loss = model.compute_loss(x_batch, h, lambda_rec)
            total_loss.backward()
            optimizer.step()
            total_loss_epoch += total_loss.item()
            rec_loss_epoch += rec_loss.item()
            nll_loss_epoch += nll_loss.item()
        avg_total_loss = total_loss_epoch / len(data_loader)
        avg_rec_loss = rec_loss_epoch / len(data_loader)
        avg_nll_loss = nll_loss_epoch / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_total_loss:.6f}, "
              f"Rec Loss: {avg_rec_loss:.6f}, NLL Loss: {avg_nll_loss:.6f}")


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    extrinsic_dim = 100  # High-dimensional data dimension
    intrinsic_dim = 10  # Latent space dimension
    autoencoder_hidden_dims = [64, 32]
    sde_hidden_dims = [32, 16]
    encoder_activation = nn.ReLU
    decoder_activation = nn.ReLU
    drift_activation = nn.Tanh
    diffusion_activation = nn.Tanh
    noise_dim = intrinsic_dim
    num_epochs = 50
    h = 1.0 / 252  # Time step size
    lambda_rec = 1.0  # Weighting parameter for reconstruction loss
    lr = 1e-3  # Learning rate
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate synthetic data for demonstration (replace with your data)
    N = 1000  # Number of samples
    n = 50  # Sequence length
    data = torch.randn(N, n, extrinsic_dim)  # Replace with your actual data

    # Create DataLoader
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = AutoencoderNeuralSDE(
        extrinsic_dim=extrinsic_dim,
        intrinsic_dim=intrinsic_dim,
        autoencoder_hidden_dims=autoencoder_hidden_dims,
        sde_hidden_dims=sde_hidden_dims,
        encoder_activation=encoder_activation,
        decoder_activation=decoder_activation,
        drift_activation=drift_activation,
        diffusion_activation=diffusion_activation,
        noise_dim=noise_dim
    )

    # Train the model
    train(model, data_loader, num_epochs, h, lambda_rec, lr, device)
