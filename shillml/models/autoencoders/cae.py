from typing import List, Tuple
from torch import Tensor
import torch.nn as nn
from shillml.models.autoencoders.ae import AutoEncoder

class CAE(AutoEncoder):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        Contractive Autoencoder (CAE) implementation, inheriting from the vanilla Autoencoder (AE) class.

        A contractive autoencoder introduces an additional regularization term to the loss function
        to encourage the learned representations to be robust to small perturbations in the input space.

        :param extrinsic_dim: the observed extrinsic high dimension
        :param intrinsic_dim: the latent intrinsic dimension
        :param hidden_dims: list of hidden dimensions for the encoder and decoder
        :param encoder_act: the encoder activation function
        :param decoder_act: the decoder activation function
        :param args: additional arguments to pass to nn.Module
        :param kwargs: additional keyword arguments to pass to nn.Module
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the contractive autoencoder.

        The forward pass not only reconstructs the input but also computes the Jacobian of the encoder
        with respect to the input, which is used in the contractive regularization term.

        :param x: the observed point cloud of shape (batch_size, extrinsic_dim)
        :return: tuple containing the reconstructed point cloud 'x_hat' of shape
                 (batch_size, extrinsic_dim) and the Jacobian 'dpi' of shape
                 (batch_size, intrinsic_dim, extrinsic_dim)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        dpi = self.encoder_jacobian(x)
        return x_hat, dpi


if __name__ == "__main__":
    import sympy as sp
    import matplotlib.pyplot as plt
    from shillml.utils import fit_model
    from shillml.losses import CAELoss
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
    ae = CAE(3, 2, [64], nn.Tanh(), nn.Tanh())
    ae_loss = CAELoss(contractive_weight=0.005)
    # Fit the model
    fit_model(ae, ae_loss, x, epochs=5000, batch_size=10)
    # Detach and plot
    x = x.detach()
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="cae")
    plt.show()