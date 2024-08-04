from shillml.models.autoencoders.ae import AE
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor


class CUCHTBAE(AE):
    """
        Curvature Contractive Hessian Tangent Bundle Autoencoder (CUCHTBAE) class that extends a standard autoencoder
        to incorporate curvature regularization given an observed drift vector field and
        infinitesimal covariance matrix field. This regularization helps in
        preserving the local geometry of the data manifold.

        Attributes:
            extrinsic_dim (int): The dimensionality of the input data.
            intrinsic_dim (int): The dimensionality of the latent space.
            encoder (FeedForwardNeuralNetwork): the FFNN representing the encoder
            decoder (FeedForwardNeuralNetwork): the FFNN representing the decoder
    """
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
            Initializes the Curvature Contractive Hessian Tangent Bundle Autoencoder (CUCHTBAE) with the specified parameters.

            Args:
                extrinsic_dim (int): The dimensionality of the input data.
                intrinsic_dim (int): The dimensionality of the latent space.
                hidden_dims (List[int]): List of dimensions for hidden layers in the encoder and decoder.
                encoder_act (nn.Module): Activation function for the encoder.
                decoder_act (nn.Module): Activation function for the decoder.
                args: Additional positional arguments for the base class initialization.
                kwargs: Additional keyword arguments for the base class initialization.
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
            Forward pass of the Curvature Contractive Hessian Tangent Bundle Autoencoder.

            Args:
                x (Tensor): Input tensor with shape (batch_size, extrinsic_dim).

            Returns: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A tuple containing:

                - x_hat (Tensor): Reconstructed input with shape (batch_size, extrinsic_dim).

                - dpi (Tensor): the Jacobian of the encoder with shape (batch_size, intrinsic_dim, extrinsic_dim)

                - model_projection (Tensor): the model orthogonal projection with shape (batch_size, extrinsic_dim, extrinsic_dim)

                - decoder_hessian (Tensor): Hessian tensor of the decoder, a tensor with
                shape (batch_size, extrinsic_dim, intrinsic_dim, extrinsic_dim)

                - encoder_hessian (Tensor): Hessian tensor of the encoder, a tensor with
                shape (batch_size, intrinsic_dim, extrinsic_dim, extrinsic_dim)

        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        dpi = self.encoder_jacobian(x)
        model_projection = self.neural_orthogonal_projection(z)
        decoder_hessian = self.decoder_hessian(z)
        encoder_hessian = self.encoder_hessian(x)
        return x_hat, dpi, model_projection, decoder_hessian, encoder_hessian


if __name__ == "__main__":
    from shillml.utils import fit_model
    from shillml.losses import CUCHTBAELoss
    import sympy as sp
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.utils import process_data
    import matplotlib.pyplot as plt
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
    ae = CUCHTBAE(3, 2, [8], nn.Tanh(), nn.Tanh())
    ae_loss = CUCHTBAELoss(contractive_weight=0.005, hessian_weight=0.00001, tangent_drift_weight=0.0001,
                           tangent_bundle_weight=0.001)
    fit_model(ae, ae_loss, x, targets=(p, orthogcomp, mu, cov), epochs=5000, batch_size=20)
    # Detach and plot
    x = x.detach()
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="ctbae")
    plt.show()