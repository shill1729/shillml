from typing import Any
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from shillml.ffnn import FeedForwardNeuralNet
from shillml.sdes import SDE


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


class LatentNeuralSDE(nn.Module):
    def __init__(self,
                 intrinsic_dim: int,
                 h1: List[int],
                 h2: List[int],
                 drift_act: nn.Module,
                 diffusion_act: nn.Module,
                 encoder_act: Optional[nn.Module] = None,
                 *args,
                 **kwargs):
        """

        :param intrinsic_dim:
        :param h1:
        :param h2:
        :param drift_act:
        :param diffusion_act:
        :param encoder_act:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.intrinsic_dim = intrinsic_dim
        neurons_mu = [intrinsic_dim] + h1 + [intrinsic_dim]
        neurons_sigma = [intrinsic_dim] + h2 + [intrinsic_dim ** 2]
        activations_mu = [drift_act for _ in range(len(neurons_mu) - 2)] + [encoder_act]
        activations_sigma = [diffusion_act for _ in range(len(neurons_sigma) - 2)] + [encoder_act]
        self.drift_net = FeedForwardNeuralNet(neurons_mu, activations_mu)
        self.diffusion_net = FeedForwardNeuralNet(neurons_sigma, activations_sigma)

    def diffusion(self, z: Tensor) -> Tensor:
        """

        :param z:
        :return:
        """
        return self.diffusion_net(z).view((z.size(0), self.intrinsic_dim, self.intrinsic_dim))

    def latent_drift_fit(self, t: float, z: np.ndarray) -> np.ndarray:
        """ For numpy EM SDE solvers"""
        with torch.no_grad():
            return self.drift_net(torch.tensor(z, dtype=torch.float32)).detach().numpy()

    def latent_diffusion_fit(self, t: float, z: np.ndarray) -> np.ndarray:
        """ For numpy EM SDE solvers"""
        d = z.shape[0]
        with torch.no_grad():
            return self.diffusion_net(torch.tensor(z, dtype=torch.float32)).view((d, d)).detach().numpy()

    def sample_paths(self, z0: np.ndarray, tn: float, ntime: int, npaths: int) -> np.ndarray:
        """

        :param z0:
        :param tn:
        :param ntime:
        :param npaths:
        :return:
        """
        # Initialize SDE object
        latent_sde = SDE(self.latent_drift_fit, self.latent_diffusion_fit)
        # Generate sample ensemble
        latent_ensemble = latent_sde.sample_ensemble(z0, tn, ntime, npaths, noise_dim=self.intrinsic_dim)
        return latent_ensemble


class AutoEncoderDiffusionGeometry(nn.Module):
    def __init__(self,
                 latent_sde: LatentNeuralSDE,
                 ae: AutoEncoder,
                 *args,
                 **kwargs):
        """

        :param extrinsic_dim:
        :param intrinsic_dim:
        :param hidden_dims: [h1, h2, h3] where each h is a list of ints for hidden dimensions of AE, and drift,
        diffusion
        :param activations: [encoder_act, decoder_act, drift_act, diffusion_act]
        :param sde_final_act: optional final activation for drift and diffusion
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.observed_projection = None
        self.observed_ambient_drift = None

        self.latent_sde = latent_sde
        self.autoencoder = ae
        self.intrinsic_dim = ae.intrinsic_dim
        self.extrinsic_dim = ae.extrinsic_dim

    def lift_sample_paths(self, latent_ensemble) -> np.ndarray:
        # Lift the latent paths to the ambient space
        lifted_ensemble = np.array([self.autoencoder.decoder(torch.tensor(path, dtype=torch.float32)).detach().numpy()
                                    for path in latent_ensemble])
        return lifted_ensemble

    def ambient_quadratic_variation_drift(self, latent_covariance: Tensor, decoder_hessian: Tensor) -> Tensor:
        qv = torch.stack(
            [torch.einsum("nii -> n", torch.bmm(latent_covariance, decoder_hessian[:, i, :, :])) for i in
             range(self.extrinsic_dim)])
        qv = qv.T
        return qv

    def compute_sde_manifold_tensors(self, x: Tensor):
        z = self.autoencoder.encoder(x)
        dphi = self.autoencoder.decoder_jacobian(z)
        latent_diffusion = self.latent_sde.diffusion(z)
        latent_covariance = torch.bmm(latent_diffusion, latent_diffusion.mT)
        hessian = self.autoencoder.decoder_hessian(z)
        q = self.ambient_quadratic_variation_drift(latent_covariance, hessian)
        return z, dphi, latent_diffusion, q


class AutoEncoderDiffusion(AutoEncoderDiffusionGeometry):

    def __init__(self, latent_sde: LatentNeuralSDE, ae: AutoEncoder, *args, **kwargs):
        super().__init__(latent_sde, ae, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x)
        bbt = torch.bmm(b, b.mT)
        # ambient_diffusion = torch.bmm(dphi, b)
        # cov_model = torch.bmm(ambient_diffusion, ambient_diffusion.mT)
        cov_model = torch.bmm(torch.bmm(dphi, bbt), dphi.mT)
        # Normal Bundle Penalty
        g = torch.bmm(dphi.mT, dphi)
        g_inv = torch.linalg.inv(g)
        P = torch.bmm(torch.bmm(dphi, g_inv), dphi.mT)
        N = torch.eye(self.extrinsic_dim).expand(x.shape[0], self.extrinsic_dim, self.extrinsic_dim) - P
        # Exact normal term
        return cov_model, N, q, bbt


class AutoEncoderDrift(AutoEncoderDiffusionGeometry):

    def __init__(self, latent_sde: LatentNeuralSDE, ae: AutoEncoder, *args, **kwargs):
        super().__init__(latent_sde, ae, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x)
        a = self.latent_sde.drift_net(z)
        tangent_drift = torch.bmm(dphi, a.unsqueeze(2)).squeeze()
        mu_model = tangent_drift + 0.5 * q
        return mu_model


def fit_model(model: nn.Module,
              loss: nn.Module,
              input_data: Tensor,
              targets: Any,
              lr: float = 0.001, epochs: int = 1000,
              print_freq: int = 1000, weight_decay: float = 0.) -> None:
    """
    Trains the given model using the specified loss function and data.
    Assumes the input of the loss function is (output, targets, regs) with regs optional, default to None.
    'output' is the output of the model.

    Args:
        model (nn.Module): The neural network model to be trained.
        loss (Any): The loss function used for training.
        input_data (Tensor): input data to the network model
        targets (Tensor): labels/targets data for the network output
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of epochs to train the model. Defaults to 1000.
        print_freq (int, optional): Frequency of printing the training loss. Defaults to 1000.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 0.

    Returns:
        None
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        output = model(input_data)
        total_loss = loss(output, targets)
        total_loss.backward()
        optimizer.step()
        if epoch % print_freq == 0:
            print('Epoch: {}: Train-Loss: {}:'.format(epoch, total_loss.item()))
    return None


class ContractiveRegularization(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(encoder_jacobian: Tensor) -> Tensor:
        """
        Computes the contractive regularization penalty from the encoder Jacobian.

        Args:
            encoder_jacobian (Tensor): Jacobian matrix of the encoder.

        Returns:
            Tensor: Contractive regularization penalty.
        """
        encoder_jacobian_norm = torch.linalg.matrix_norm(encoder_jacobian, ord="fro")
        contraction_penalty = torch.mean(encoder_jacobian_norm ** 2)
        return contraction_penalty


class TangentBundleLoss(nn.Module):
    def __init__(self, norm="fro", *args, **kwargs):
        self.norm = norm
        super().__init__(*args, **kwargs)

    def forward(self, model_projection: Tensor, observed_projection: Tensor) -> Tensor:
        """
        Computes the tangent bundle loss between observed and model projections.

        Args:
            model_projection (Tensor): Model projection matrix.
            observed_projection (Tensor): Observed projection matrix.

        Returns:
            Tensor: Tangent bundle loss.
        """
        orthogonal_projection_error = torch.linalg.matrix_norm(model_projection - observed_projection, ord=self.norm)
        tangent_bundle_error = torch.mean(orthogonal_projection_error ** 2)
        return tangent_bundle_error


class CAELoss(nn.Module):
    def __init__(self, contractive_weight=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contractive_weight = contractive_weight
        self.mse_loss = nn.MSELoss()
        self.contractive_reg = ContractiveRegularization()

    def forward(self, output, x):
        x_hat, dpi = output
        total_loss = self.mse_loss(x_hat, x) + self.contractive_weight * self.contractive_reg(dpi)
        return total_loss


class TBAELoss(nn.Module):
    def __init__(self, tangent_bundle_weight=1., observed_projection=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tangent_bundle_weight = tangent_bundle_weight
        self.mse_loss = nn.MSELoss()
        self.tangent_bundle_reg = TangentBundleLoss()
        self.observed_projection = observed_projection

    def forward(self, output, x):
        x_hat, model_projection = output
        total_loss = (self.mse_loss(x_hat, x) +
                      self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection))
        return total_loss


class CTBAELoss(nn.Module):
    def __init__(self, contractive_weight=1., tangent_bundle_weight=1., observed_projection=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contractive_weight = contractive_weight
        self.tangent_bundle_weight = tangent_bundle_weight
        self.mse_loss = nn.MSELoss()
        self.tangent_bundle_reg = TangentBundleLoss()
        self.contractive_reg = ContractiveRegularization()
        self.observed_projection = observed_projection

    def forward(self, output, x):
        x_hat, dpi, model_projection = output
        mse = self.mse_loss(x_hat, x)
        contr = self.contractive_reg(dpi) * self.contractive_weight
        tang = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection)
        total_loss = mse + contr + tang
        return total_loss


class CovarianceMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(model_cov: Tensor, observed_cov: Tensor) -> Tensor:
        """
        Computes the mean squared error (MSE) between observed and model covariance matrices.

        Args:
            model_cov (Tensor): Model covariance matrix.
            observed_cov (Tensor): Observed covariance matrix.

        Returns:
            Tensor: Mean squared error between observed and model covariance matrices.
        """
        cov_error = torch.linalg.matrix_norm(model_cov - observed_cov, ord="fro")
        mse_cov = torch.mean(cov_error ** 2)
        return mse_cov


class NormalBundleLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(normal_projected_tangent_vector: Tensor) -> Tensor:
        """
        Computes the normal bundle loss for the normal projected tangent vector.

        Args:
            normal_projected_tangent_vector (Tensor): Normal projected tangent vector.

        Returns:
            Tensor: Normal bundle loss.
        """
        normal_penalty = torch.mean(torch.linalg.vector_norm(normal_projected_tangent_vector, ord=2) ** 2)
        return normal_penalty


class DiffusionLoss(nn.Module):
    def __init__(self, normal_bundle_weight=0.0):
        super().__init__()
        self.normal_bundle_weight = normal_bundle_weight
        self.cov_mse = CovarianceMSELoss()
        self.local_cov_mse = CovarianceMSELoss()
        self.normal_bundle_loss = NormalBundleLoss()

    def forward(self, model_output, targets):
        """

        :param model_output: (model_cov, normal_proj, qv, bbt)
        :param targets: (observed_cov, ambient_drift, encoded_observed_cov)
        :return:
        """
        model_cov, normal_proj, qv, bbt = model_output
        observed_cov, ambient_drift, encoded_observed_cov = targets
        tangent_vector = ambient_drift - 0.5 * qv
        normal_proj_vector = torch.bmm(normal_proj, tangent_vector.unsqueeze(2))
        cov_mse = self.cov_mse(model_cov, observed_cov)
        # Add local cov mse
        local_cov_mse = self.local_cov_mse(bbt, encoded_observed_cov)
        normal_bundle_loss = self.normal_bundle_loss(normal_proj_vector)
        total_loss = local_cov_mse + self.normal_bundle_weight * normal_bundle_loss
        return total_loss

    def extra_repr(self) -> str:
        return f'contractive_weight={self.contractive_weight}, tangent_bundle_weight={self.tangent_bundle_weight}'


class DriftMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(model_ambient_drift: Tensor, observed_ambient_drift: Tensor) -> Tensor:
        """
        Computes the mean squared error (MSE) between observed and model ambient drift vectors.

        Args:
            model_ambient_drift (Tensor): Model ambient drift vector.
            observed_ambient_drift (Tensor): Observed ambient drift vector.

        Returns:
            Tensor: Mean squared error between observed and model ambient drift vectors.
        """
        return torch.mean(torch.linalg.vector_norm(model_ambient_drift - observed_ambient_drift, ord=2, dim=1) ** 2)
