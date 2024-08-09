from typing import List
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from shillml.models.ffnn import FeedForwardNeuralNet


class AE(nn.Module):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        A vanilla Auto-encoder using FeedForwardNeuralNet for encoding and decoding.

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
        # Encoder and decoder architecture
        encoder_neurons = [extrinsic_dim] + hidden_dims + [intrinsic_dim]
        decoder_neurons = encoder_neurons[::-1]
        encoder_acts = [encoder_act] * (len(hidden_dims) + 1)
        decoder_acts = [decoder_act] * len(hidden_dims) + [None]
        self.encoder = FeedForwardNeuralNet(encoder_neurons, encoder_acts)
        self.decoder = FeedForwardNeuralNet(decoder_neurons, decoder_acts)
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

    def compute_diffeo_error(self, x: Tensor) -> Tensor:
        """
        Compute the diffeomorphism error for a batch of inputs.

        The error is calculated as ||D_pi * D_phi - I||_F^2, where:
        D_pi is the encoder Jacobian
        D_phi is the decoder Jacobian
        I is the identity matrix
        ||.||_F is the Frobenius norm

        :param x: Input tensor of shape (n, d) where n is the batch size and d is the extrinsic dimension
        :return: Average diffeomorphism error over the batch
        """
        # Compute encoder Jacobian (D_pi)
        D_pi = self.encoder_jacobian(x)

        # Encode the input to get latent representations
        z = self.encoder(x)

        # Compute decoder Jacobian (D_phi)
        D_phi = self.decoder_jacobian(z)

        # Compute D_pi * D_phi
        composition = torch.bmm(D_pi, D_phi)

        # Create identity matrix of appropriate size
        I = torch.eye(self.intrinsic_dim).expand(x.size(0), self.intrinsic_dim, self.intrinsic_dim)

        # Compute the error: ||D_pi * D_phi - I||_F^2
        error = torch.linalg.matrix_norm(composition - I, ord='fro') ** 2

        # Return the average error over the batch
        return torch.mean(error)

    def plot_surface(self, a: float, b: float, grid_size: int, ax=None, title=None) -> None:
        """
        Plot the surface produced by the neural-network chart.

        :param title:
        :param a: the lb of the encoder range box [a,b]^d
        :param b: the ub of the encoder range box [a,b]^d
        :param grid_size: grid size for the mesh of the encoder range
        :param ax: plot axis object
        :return:
        """
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
        return None


if __name__ == "__main__":
    import sympy as sp
    from shillml.utils import fit_model
    from shillml.losses import AELoss
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.utils import process_data
    # Generate data
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])
    bounds = [(-1, 1), (-1, 1)]
    c1 = 1
    c2 = 1
    chart = sp.Matrix([u, v, (u / c1) ** 2 + (v / c2) ** 2])
    manifold = RiemannianManifold(local_coordinates, chart)
    local_drift = manifold.local_bm_drift()
    local_diffusion = manifold.local_bm_diffusion()
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion)
    x, _, mu, cov, _ = cloud.generate(30)
    x, mu, cov, p, orthogcomp = process_data(x, mu, cov, d=2)
    # Define model
    ae = AE(3, 2, [64], nn.Tanh(), nn.Tanh())
    ae_loss = AELoss()
    # Fit the model
    fit_model(ae, ae_loss, x, epochs=5000, batch_size=10)
    # Detach and plot
    x = x.detach()
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="ae")
    plt.show()


