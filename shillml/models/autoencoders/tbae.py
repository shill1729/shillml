from shillml.models.autoencoders.ae import AE
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor


class TBAE(AE):
    """
        Tangent Bundle Autoencoder (TBAE) class that extends a standard autoencoder
        to incorporate tangent bundle regularization. This regularization helps in
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
        Initializes the Tangent Bundle Autoencoder (TBAE) with the specified parameters.

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Tangent Bundle Autoencoder.

        Args:
            x (Tensor): Input tensor with shape (batch_size, extrinsic_dim).

        Returns: Tuple[Tensor, Tensor]: A tuple containing: - x_hat (Tensor): Reconstructed input with shape (
        batch_size, extrinsic_dim). - model_projection (Tensor): Projection of the latent representation on the
        tangent bundle with shape (batch_size, extrinsic_dim, extrinsic_dim).
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        model_projection = self.neural_orthogonal_projection(z)
        return x_hat, model_projection
