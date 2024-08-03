from shillml.models.autoencoders.ae import AE
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor


class CUCTBAE(AE):
    """
        Curvature Contractive Tangent Bundle Autoencoder (CUCTBAE) class that extends a standard autoencoder
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
            Initializes the Curvature Contractive Tangent Bundle Autoencoder (CUCTBAE) with the specified parameters.

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
            Forward pass of the Curvature Contractive Tangent Bundle Autoencoder.

            Args:
                x (Tensor): Input tensor with shape (batch_size, extrinsic_dim).

            Returns: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A tuple containing:

                - x_hat (Tensor): Reconstructed input with shape (batch_size, extrinsic_dim).

                - dpi (Tensor): the Jacobian of the encoder with shape (batch_size, intrinsic_dim, extrinsic_dim)

                - model_projection (Tensor): the model orthogonal projection with shape (batch_size, extrinsic_dim, extrinsic_dim)

                - decoder_hessian (Tensor): Hessian tensor of the decoder, a tensor with
                shape (batch_size, extrinsic_dim, intrinsic_dim, extrinsic_dim)


        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        dpi = self.encoder_jacobian(x)
        model_projection = self.neural_orthogonal_projection(z)
        decoder_hessian = self.decoder_hessian(z)
        return x_hat, dpi, model_projection, decoder_hessian
