from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


from shillml.models.autoencoders import AE, CAE, CHAE, TBAE, CTBAE, CUCTBAE, CUCHTBAE
from shillml.models.nsdes import AutoEncoderDiffusion


def contractive_regularization(encoder_jacobian: Tensor) -> Tensor:
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


def hessian_regularization(hessian: torch.Tensor) -> torch.Tensor:
    """
    Computes the encoder Hessian regularization penalty for a batch of points.

    Args:
        hessian (Tensor): Hessian tensor of shape (n, d, D, D).

    Returns:
        Tensor: Hessian regularization penalty.
    """
    # Compute the Frobenius norm of the Hessian for each point and output dimension
    hessian_frobenius_norm = torch.linalg.matrix_norm(hessian, ord='fro', dim=(2, 3))
    penalty = torch.sum(hessian_frobenius_norm ** 2, dim=1)  # Sum over the output dimensions
    return torch.mean(penalty)  # Average over the batch


def tangent_drift_loss(normal_projected_tangent_vector: Tensor) -> Tensor:
    """
    Computes the normal bundle loss for the normal projected tangent vector.

    Args:
        normal_projected_tangent_vector (Tensor): Normal projected tangent vector.

    Returns:
        Tensor: Normal bundle loss.
    """
    normal_penalty = torch.mean(torch.linalg.vector_norm(normal_projected_tangent_vector, ord=2) ** 2)
    return normal_penalty


class MatrixMSELoss(nn.Module):
    def __init__(self, norm="fro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, input_data: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.linalg.matrix_norm(input_data - target, ord=self.norm) ** 2)


class AELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, ae: AE, x: Tensor, drift: Optional[Tensor] = None, cov: Optional[Tensor] = None):
        x_hat = ae.forward(x)
        return self.reconstruction_loss(x_hat, x)


class CAELoss(AELoss):
    def __init__(self, contractive_weight=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contractive_weight = contractive_weight
        self.contractive_reg = contractive_regularization

    def forward(self, cae: CAE, x: Tensor, drift: Optional[Tensor] = None, cov: Optional[Tensor] = None):
        x_hat, dpi = cae.forward(x)
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        contractive_penalty = self.contractive_weight * self.contractive_reg(dpi)
        total_loss = reconstruction_loss + contractive_penalty
        return total_loss


class CHAELoss(CAELoss):
    def __init__(self, contractive_weight=1., hessian_weight=1., *args, **kwargs):
        super().__init__(contractive_weight, *args, **kwargs)
        self.hessian_weight = hessian_weight
        self.hessian_reg = hessian_regularization

    def forward(self, chae: CHAE, x: Tensor, drift: Optional[Tensor] = None, cov: Optional[Tensor] = None):
        x_hat, dpi, hessian_pi = chae.forward(x)
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        contractive_loss = self.contractive_weight * self.contractive_reg(dpi)
        hessian_loss = self.hessian_weight * self.hessian_reg(hessian_pi)
        total_loss = reconstruction_loss + contractive_loss + hessian_loss
        return total_loss


class TBAELoss(AELoss):
    def __init__(self,
                 observed_projection=None,
                 tangent_bundle_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if observed_projection is None:
            raise ValueError("'observed_projection' cannot be 'None'.")
        self.tangent_bundle_weight = tangent_bundle_weight
        self.tangent_bundle_reg = MatrixMSELoss(norm=norm)
        self.observed_projection = observed_projection

    def forward(self, tbae: TBAE, x: Tensor, drift: Optional[Tensor] = None, cov: Optional[Tensor] = None):
        x_hat, model_projection = tbae.forward(x)
        reconstruction_mse = self.reconstruction_loss(x_hat, x)
        tangent_mse = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection)
        total_loss = reconstruction_mse + tangent_mse
        return total_loss


class CTBAELoss(TBAELoss, CAELoss):
    def __init__(self,
                 observed_projection=None,
                 tangent_bundle_weight=1.,
                 contractive_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        TBAELoss.__init__(self, observed_projection, tangent_bundle_weight, norm, *args, **kwargs)
        CAELoss.__init__(self, contractive_weight, *args, **kwargs)

    def forward(self, ctbae: CTBAE, x: Tensor, drift: Optional[Tensor] = None, cov: Optional[Tensor] = None):
        x_hat, dpi, model_projection = ctbae.forward(x)
        mse = self.reconstruction_loss(x_hat, x)
        contractive_penalty = self.contractive_reg(dpi) * self.contractive_weight
        tangent_error = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection)
        total_loss = mse + contractive_penalty + tangent_error
        return total_loss


class CUCTBAELoss(CTBAELoss):
    def __init__(self,
                 tangent_drift_weight=1.,
                 observed_projection=None,
                 tangent_bundle_weight=1.,
                 contractive_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        super().__init__(observed_projection, tangent_bundle_weight, contractive_weight, norm, *args, **kwargs)
        self.curvature_weight = tangent_drift_weight
        self.tangent_drift_loss = tangent_drift_loss
        n, D, _ = self.observed_projection.size()
        self.extrinsic_dim = D
        self.observed_normal_projection = torch.eye(D).expand(n, D, D) - self.observed_projection

    def forward(self, cuctbae: CUCTBAE, x: Tensor, drift: Optional[Tensor] = None, cov: Optional[Tensor] = None):
        x_hat, dpi, model_projection, decoder_hessian = cuctbae.forward(x)
        # x, ambient_drift, ambient_cov, observed_proj = targets
        bbt_proxy = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
        qv = torch.stack(
            [torch.einsum("nii -> n", torch.bmm(bbt_proxy, decoder_hessian[:, i, :, :])) for i in
             range(self.extrinsic_dim)])
        qv = qv.T
        tangent_vector = drift - 0.5 * qv
        normal_proj_vector = torch.bmm(self.observed_normal_projection, tangent_vector.unsqueeze(2))
        mse = self.reconstruction_loss(x_hat, x)
        contractive_penalty = self.contractive_reg(dpi) * self.contractive_weight
        tangent_bundle_error = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection,
                                                                                    self.observed_projection)
        tangent_drift_error = self.curvature_weight * self.tangent_drift_loss(normal_proj_vector)
        total_loss = mse + contractive_penalty + tangent_bundle_error + tangent_drift_error
        return total_loss


class CUCHTBAELoss(CUCTBAELoss, CHAELoss):
    def __init__(self,
                 contractive_weight=1.,
                 hessian_weight=1.,
                 tangent_bundle_weight=1.,
                 tangent_drift_weight=1.,
                 observed_projection=None,
                 norm="fro",
                 *args,
                 **kwargs):
        CHAELoss.__init__(self, contractive_weight, hessian_weight, *args, **kwargs)
        CUCTBAELoss.__init__(self, tangent_drift_weight, observed_projection, tangent_bundle_weight, contractive_weight, norm, *args, **kwargs)

    def forward(self, cuchtbae: CUCHTBAE, x: Tensor, drift: Optional[Tensor] = None, cov: Optional[Tensor] = None):
        x_hat, dpi, model_projection, decoder_hessian, encoder_hessian = cuchtbae.forward(x)
        bbt_proxy = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
        qv = torch.stack(
            [torch.einsum("nii -> n", torch.bmm(bbt_proxy, decoder_hessian[:, i, :, :])) for i in
             range(self.extrinsic_dim)])
        qv = qv.T
        tangent_vector = drift - 0.5 * qv
        normal_proj_vector = torch.bmm(self.observed_normal_projection, tangent_vector.unsqueeze(2))
        mse = self.reconstruction_loss(x_hat, x)
        contr = self.contractive_reg(dpi) * self.contractive_weight
        hess = self.hessian_weight * self.hessian_reg(encoder_hessian)
        tang = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection)
        curv = self.tangent_drift_weight * self.tangent_drift_loss(normal_proj_vector)
        total_loss = mse + contr + tang + curv + hess
        return total_loss


class DiffusionLoss(nn.Module):
    def __init__(self, tangent_drift_weight=0.0, norm="fro"):
        super().__init__()
        self.tangent_drift_weight = tangent_drift_weight
        self.cov_mse = MatrixMSELoss(norm=norm)
        self.local_cov_mse = MatrixMSELoss(norm=norm)
        self.tangent_drift_loss = tangent_drift_loss

    def forward(self, ae_diffusion: AutoEncoderDiffusion, x, ambient_drift, ambient_cov, encoded_cov):
        model_cov, normal_proj, qv, bbt = ae_diffusion.forward(x)
        tangent_vector = ambient_drift - 0.5 * qv
        normal_proj_vector = torch.bmm(normal_proj, tangent_vector.unsqueeze(2))
        cov_mse = self.cov_mse(model_cov, ambient_cov)
        # Add local cov mse
        local_cov_mse = self.local_cov_mse(bbt, encoded_cov)
        normal_bundle_loss = self.normal_bundle_loss(normal_proj_vector)
        total_loss = cov_mse + local_cov_mse + self.tangent_drift_weight * normal_bundle_loss
        return total_loss
