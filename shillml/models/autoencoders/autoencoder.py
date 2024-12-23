"""
    NEW!

    An implementation of an auto-encoder using PyTorch. It is implemented as a class with various methods
    for computing objects from differential geometry (e.g. orthogonal projections) as well as having
    loss functions and penalties as methods.
"""
from typing import List
from torch import Tensor

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch

from shillml.models.ffnn import FeedForwardNeuralNet


class AutoEncoder1(nn.Module):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        An Auto-encoder using FeedForwardNeuralNet for encoding and decoding.

        Many methods are provided for differential geometry computations.

        :param extrinsic_dim: the observed extrinsic high dimension
        :param intrinsic_dim: the latent intrinsic dimension
        :param hidden_dims: list of hidden dimensions for the encoder and decoder
        :param encode_act: the encoder activation function
        :param decode_act: the decoder activation function
        :param args: args to pass to nn.Module
        :param kwargs: kwargs to pass to nn.Module
        """
        super().__init__(*args, **kwargs)
        self.intrinsic_dim = intrinsic_dim
        self.extrinsic_dim = extrinsic_dim
        # Encoder and decoder architecture:
        encoder_neurons = [extrinsic_dim] + hidden_dims + [intrinsic_dim]
        # The decoder's layer structure is the reverse of the encoder
        decoder_neurons = encoder_neurons[::-1]
        encoder_acts = [encoder_act] * (len(hidden_dims) + 1)
        # The decoder has no final activation, so it can target anything in the ambient space
        decoder_acts = [decoder_act] * len(hidden_dims) + [None]
        self.encoder = FeedForwardNeuralNet(encoder_neurons, encoder_acts)
        self.decoder = FeedForwardNeuralNet(decoder_neurons, decoder_acts)
        # Tie the weights of the decoder to be the transpose of the encoder, in reverse due
        self.decoder.tie_weights(self.encoder)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the autoencoder.

        :param x: the observed point cloud of shape (batch_size, extrinsic_dim)
        :return: the reconstructed point cloud x_hat of shape
                 (batch_size, extrinsic_dim)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encoder_jacobian(self, x: Tensor) -> Tensor:
        """
        Compute the Jacobian of the encoder at the given input.

        :param x: Input tensor of shape (batch_size, extrinsic_dim)
        :return: Jacobian matrix of shape (batch_size, intrinsic_dim, extrinsic_dim)
        """
        return self.encoder.jacobian_network(x)

    def decoder_jacobian(self, z: Tensor) -> Tensor:
        """
        Compute the Jacobian of the decoder at the given latent representation.

        :param z: Latent tensor of shape (batch_size, intrinsic_dim)
        :return: Jacobian matrix of shape (batch_size, extrinsic_dim, intrinsic_dim)
        """
        return self.decoder.jacobian_network(z)

    def decoder_hessian(self, z: Tensor) -> Tensor:
        """
        Compute the Hessian of the decoder at the given latent representation.

        :param z: Latent tensor of shape (batch_size, intrinsic_dim)
        :return: Hessian matrix of shape (batch_size, extrinsic_dim, intrinsic_dim, intrinsic_dim)
        """
        return self.decoder.hessian_network(z)

    def encoder_hessian(self, x: Tensor) -> Tensor:
        """
        Compute the Hessian of the encoder at the given input.

        :param x: Input tensor of shape (batch_size, extrinsic_dim)
        :return: Hessian matrix of shape (batch_size, intrinsic_dim, extrinsic_dim, extrinsic_dim)
        """
        return self.encoder.hessian_network(x)

    def neural_orthogonal_projection(self, z: Tensor) -> Tensor:
        """
            Compute the orthogonal projection onto the tangent space of the decoder at z.

            :param z: Latent tensor of shape (batch_size, intrinsic_dim)
            :return: Orthogonal projection matrix of shape
                     (batch_size, extrinsic_dim, extrinsic_dim)
        """
        dphi = self.decoder_jacobian(z)
        g = torch.bmm(dphi.mT, dphi)
        g_inv = torch.linalg.inv(g)
        P = torch.bmm(torch.bmm(dphi, g_inv), dphi.mT)
        return P

    def neural_metric_tensor(self, z: Tensor) -> Tensor:
        """
            Compute the Riemannian metric tensor induced by the decoder at z.

            :param z: Latent tensor of shape (batch_size, intrinsic_dim)
            :return: Metric tensor of shape (batch_size, intrinsic_dim, intrinsic_dim)
        """
        dphi = self.decoder_jacobian(z)
        g = torch.bmm(dphi.mT, dphi)
        return g

    def neural_volume_measure(self, z: Tensor) -> Tensor:
        """
            Compute the volume measure induced by the Riemannian metric tensor.

            :param z: Latent tensor of shape (batch_size, intrinsic_dim)
            :return: Volume measure tensor of shape (batch_size,)
        """
        g = self.metric_tensor(z)
        return torch.sqrt(torch.linalg.det(g))

    def lift_sample_paths(self, latent_ensemble: np.ndarray) -> np.ndarray:
        """
        Lift the latent paths to the ambient space using the decoder.

        :param latent_ensemble: An array of latent paths of shape
                                (num_samples, path_length, intrinsic_dim)
        :return: Lifted ensemble in the ambient space of shape
                 (num_samples, path_length, extrinsic_dim)
        """
        lifted_ensemble = np.array([self.decoder(torch.tensor(path, dtype=torch.float32)).detach().numpy()
                                    for path in latent_ensemble])
        return lifted_ensemble

    def plot_surface(self, a: float, b: float, grid_size: int, ax=None, title=None, dim=3) -> None:
        """
        Plot the surface produced by the neural-network chart.

        :param title:
        :param a: the lb of the encoder range box [a,b]^d
        :param b: the ub of the encoder range box [a,b]^d
        :param grid_size: grid size for the mesh of the encoder range
        :param ax: plot axis object
        :param dim: dimension to plot in, default 3
        :return:
        """
        if dim == 3:
            ux = np.linspace(a, b, grid_size)
            vy = np.linspace(a, b, grid_size)
            u, v = np.meshgrid(ux, vy, indexing="ij")
            x1 = np.zeros((grid_size, grid_size))
            x2 = np.zeros((grid_size, grid_size))
            x3 = np.zeros((grid_size, grid_size))
            for i in range(grid_size):
                for j in range(grid_size):
                    x0 = np.column_stack([u[i, j], v[i, j]])
                    x0 = torch.tensor(x0, dtype=torch.float32)
                    xx = self.decoder(x0).detach().numpy()
                    x1[i, j] = xx[0, 0]
                    x2[i, j] = xx[0, 1]
                    x3[i, j] = xx[0, 2]
            if ax is not None:
                ax.plot_surface(x1, x2, x3, alpha=0.5, cmap="magma")
                if title is not None:
                    ax.set_title(title)
                else:
                    ax.set_title("NN manifold")
            else:
                raise ValueError("'ax' cannot be None")
        elif dim == 2:
            u = np.linspace(a, b, grid_size)
            x1 = np.zeros((grid_size, grid_size))
            x2 = np.zeros((grid_size, grid_size))
            for i in range(grid_size):
                x0 = np.column_stack([u[i]])
                x0 = torch.tensor(x0, dtype=torch.float32)
                xx = self.decoder(x0).detach().numpy()
                x1[i] = xx[0, 0]
                x2[i] = xx[0, 1]
            if ax is not None:
                ax.plot(x1, x2, alpha=0.9)
                if title is not None:
                    ax.set_title(title)
                else:
                    ax.set_title("NN manifold")
        return None


if __name__ == "__main__":
    import sympy as sp
    from shillml.utils import fit_model
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.utils import process_data
    from shillml.losses.loss_modules import TotalLoss, LossWeights
    from shillml.utils import compute_test_losses

    seed = 17
    torch.manual_seed(seed)
    num_pts = 30
    num_test = 1000
    batch_size = 30
    epochs = 20000
    hidden_dims = [32, 32]
    lr = 0.001
    epsilon = 0.2
    npaths = 2
    tn = 10
    ntime = 9000
    a = -1
    b = 1
    weights = LossWeights()
    weights.reconstruction = 1.
    weights.tangent_drift_weight = 0.
    weights.diffeomorphism_reg1 = 0.
    weights.encoder_contraction_weight = 0.
    bounds = [(a, b), (a, b)]
    large_bounds = [(a - epsilon, b + epsilon), (a-epsilon, b + epsilon)]
    # Generate data
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])
    fuv = (u/2)**2+(v/2)**2
    chart = sp.Matrix([u, v, fuv])

    print("Computing geometry...")
    manifold = RiemannianManifold(local_coordinates, chart)
    # The local dynamics
    print("Computing drift...")
    local_drift = sp.Matrix([0,0])
    print("Computing diffusion...")
    local_diffusion = sp.eye(2,2)

    # Generating the point cloud
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion)
    x, _, mu, cov, _ = cloud.generate(num_pts, seed=seed)
    x, mu, cov, p, orthogcomp, frame = process_data(x, mu, cov, d=2, return_frame=True)
    # Define model
    ae = AutoEncoder1(3, 2, hidden_dims, nn.Tanh(), nn.Tanh())
    ae_loss = TotalLoss(weights)
    # Print results
    pre_train_losses = compute_test_losses(ae, ae_loss, x, p, frame, cov, mu)
    print("\n Pre-training losses")
    for key, value in pre_train_losses.items():
        print(f"{key} = {value:.4f}")

    # Fit the model
    fit_model(ae, ae_loss, x, (p, frame, cov, mu), lr=lr, epochs=epochs, batch_size=batch_size)
    # Test data:
    cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion)
    x_test, _, mu_test, cov_test, _ = cloud.generate(num_test, seed=None)
    x_test, mu_test, cov_test, p_test, orthogcomp_test, frame_test = process_data(x_test, mu_test, cov_test, d=2,
                                                                                  return_frame=True)
    # Print results
    test_losses = compute_test_losses(ae, ae_loss, x_test, p_test, frame_test, cov_test, mu_test)
    print("\n Test losses")
    for key, value in test_losses.items():
        print(f"{key} = {value:.4f}")

    # Detach and plot
    x = x.detach()
    x_test = x_test.detach()

    # Create a single figure with two subplots side by side
    fig = plt.figure(figsize=(20, 10))

    # First subplot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2])
    ae.plot_surface(-1, 1, grid_size=30, ax=ax1, title="ae - Training point cloud")

    # Second subplot
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2])
    ae.plot_surface(-1, 1, grid_size=30, ax=ax2, title="ae - Test point cloud")

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

