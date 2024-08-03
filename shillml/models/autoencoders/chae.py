from shillml.models.autoencoders.ae import AE
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor


class CHAE(AE):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        Contractive Hessian Autoencoder (CHAE) implementation, inheriting from the vanilla Autoencoder (AE) class.

        A contractive Hessian autoencoder introduces a second-order regularization term to the loss function,
        encouraging the learned representations to be robust to small perturbations in the input space,
        while also considering the curvature of the encoder function.

        :param extrinsic_dim: the observed extrinsic high dimension
        :param intrinsic_dim: the latent intrinsic dimension
        :param hidden_dims: list of hidden dimensions for the encoder and decoder
        :param encoder_act: the encoder activation function
        :param decoder_act: the decoder activation function
        :param args: additional arguments to pass to nn.Module
        :param kwargs: additional keyword arguments to pass to nn.Module
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the contractive Hessian autoencoder.

        The forward pass not only reconstructs the input and computes the Jacobian of the encoder with respect to the
        input, but also computes the Hessian of the encoder, which is used in the second-order contractive
        regularization term.

        :param x: the observed point cloud of shape (batch_size, extrinsic_dim) :return: tuple containing the
        reconstructed point cloud 'x_hat' of shape (batch_size, extrinsic_dim), the Jacobian 'dpi' of shape (
        batch_size, intrinsic_dim, extrinsic_dim), and the encoder Hessian 'encoder_hessian' of shape (batch_size,
        intrinsic_dim, extrinsic_dim, extrinsic_dim)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        dpi = self.encoder_jacobian(x)
        encoder_hessian = self.encoder_hessian(x)
        return x_hat, dpi, encoder_hessian
