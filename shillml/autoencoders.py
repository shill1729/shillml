from typing import List, Tuple
from torch import Tensor

import torch.nn as nn
import numpy as np

import torch

from shillml.ffnn import FeedForwardNeuralNet


def toggle_model(model: nn.Module, on: bool = False) -> None:
    """
    Turn a nn.Module's gradient tracking on or off for its parameters
    :param model: nn.Module, the model to turn on or off
    :param on: bool, True for on, False for off
    :return: None
    """
    for param in model.parameters():
        param.requires_grad = on
    return None


class AutoEncoder(nn.Module):
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

        :param x: the observed point cloud
        :return: (x_hat, z) the reconstructed point cloud and the latent, encoded points
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def lift_sample_paths(self, latent_ensemble: np.ndarray) -> np.ndarray:
        # Lift the latent paths to the ambient space
        lifted_ensemble = np.array([self.decoder(torch.tensor(path, dtype=torch.float32)).detach().numpy()
                                    for path in latent_ensemble])
        return lifted_ensemble

    def encoder_jacobian(self, x) -> Tensor:
        return self.encoder.jacobian_network(x)

    def decoder_jacobian(self, z) -> Tensor:
        return self.decoder.jacobian_network(z)

    def decoder_hessian(self, z) -> Tensor:
        return self.decoder.hessian_network(z)

    def neural_orthogonal_projection(self, z) -> Tensor:
        dphi = self.decoder_jacobian(z)
        g = torch.bmm(dphi.mT, dphi)
        g_inv = torch.linalg.inv(g)
        P = torch.bmm(torch.bmm(dphi, g_inv), dphi.mT)
        return P

    def neural_metric_tensor(self, z) -> Tensor:
        dphi = self.decoder_jacobian(z)
        g = torch.bmm(dphi.mT, dphi)
        return g

    def neural_volume_measure(self, z) -> Tensor:
        g = self.metric_tensor(z)
        return torch.sqrt(torch.linalg.det(g))

    def plot_surface(self, a, b, grid_size, ax=None, title=None) -> None:
        """
        Plot the surface produced by the neural-network chart.

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


class ContractiveAutoEncoder(AutoEncoder):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        The forward method returns (x_hat, dpi) for contractive regularization
        :param extrinsic_dim:
        :param intrinsic_dim:
        :param hidden_dims:
        :param encoder_act:
        :param decoder_act:
        :param args:
        :param kwargs:
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        dpi = self.encoder_jacobian(x)
        return x_hat, dpi


class TangentBundleAutoEncoder(AutoEncoder):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        The forward method returns (x_hat, P_hat) for tangent bundle regularization
        :param extrinsic_dim:
        :param intrinsic_dim:
        :param hidden_dims:
        :param encoder_act:
        :param decoder_act:
        :param args:
        :param kwargs:
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        model_projection = self.neural_orthogonal_projection(z)
        return x_hat, model_projection


class ContractiveTangentBundleAutoEncoder(AutoEncoder):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        The forward method returns (x_hat, dpi, P_hat) for contractive regularization and tangent space reg
        :param extrinsic_dim:
        :param intrinsic_dim:
        :param hidden_dims:
        :param encoder_act:
        :param decoder_act:
        :param args:
        :param kwargs:
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        dpi = self.encoder_jacobian(x)
        model_projection = self.neural_orthogonal_projection(z)
        return x_hat, dpi, model_projection


class CurvatureAutoEncoder(AutoEncoder):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        The forward method returns (x_hat, Hphi) for curvature regularization

        :param extrinsic_dim:
        :param intrinsic_dim:
        :param hidden_dims:
        :param encoder_act:
        :param decoder_act:
        :param args:
        :param kwargs:
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        decoder_hessian = self.decoder_hessian(z)
        return x_hat, decoder_hessian


class CurvatureCTBAutoEncoder(AutoEncoder):
    def __init__(self,
                 extrinsic_dim: int,
                 intrinsic_dim: int,
                 hidden_dims: List[int],
                 encoder_act: nn.Module,
                 decoder_act: nn.Module,
                 *args,
                 **kwargs):
        """
        The forward method returns (x_hat, dpi, P_hat, Hphi) for contractive regularization and tangent space reg,
        and curvature regularization.

        :param extrinsic_dim:
        :param intrinsic_dim:
        :param hidden_dims:
        :param encoder_act:
        :param decoder_act:
        :param args:
        :param kwargs:
        """
        super().__init__(extrinsic_dim, intrinsic_dim, hidden_dims, encoder_act, decoder_act, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        dpi = self.encoder_jacobian(x)
        model_projection = self.neural_orthogonal_projection(z)
        decoder_hessian = self.decoder_hessian(z)
        return x_hat, dpi, model_projection, decoder_hessian
