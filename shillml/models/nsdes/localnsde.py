import torch.nn as nn
from typing import List, Optional, Tuple
import torch
import numpy as np
from torch import Tensor


from shillml.models.ffnn import FeedForwardNeuralNet
from shillml.models.autoencoders import AutoEncoder
from shillml.sdes.sdes import SDE


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
        # TODO Let's refactor this by flattening out bb^T and flattening out Hessian (phi)
        #  So that Trace(bb^T Hessian Phi) can be computed as a vector dot product
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


class AutoEncoderDiffusion2(AutoEncoderDiffusionGeometry):

    def __init__(self, latent_sde: LatentNeuralSDE, ae: AutoEncoder, *args, **kwargs):
        super().__init__(latent_sde, ae, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x)
        bbt = torch.bmm(b, b.mT)
        # Normal Bundle Penalty
        g = torch.bmm(dphi.mT, dphi)
        g_inv = torch.linalg.inv(g)
        P = torch.bmm(torch.bmm(dphi, g_inv), dphi.mT)
        N = torch.eye(self.extrinsic_dim).expand(x.shape[0], self.extrinsic_dim, self.extrinsic_dim) - P
        # Exact normal term
        return dphi, g_inv, N, q, bbt


class AutoEncoderDrift(AutoEncoderDiffusionGeometry):

    def __init__(self, latent_sde: LatentNeuralSDE, ae: AutoEncoder, *args, **kwargs):
        super().__init__(latent_sde, ae, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        z, dphi, b, q = self.compute_sde_manifold_tensors(x)
        a = self.latent_sde.drift_net(z)
        tangent_drift = torch.bmm(dphi, a.unsqueeze(2)).squeeze()
        mu_model = tangent_drift + 0.5 * q
        return mu_model
