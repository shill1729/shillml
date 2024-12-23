from typing import Optional, Any, Tuple, List, Union

import torch
import torch.nn as nn
from torch import Tensor

from shillml.models.autoencoders import AutoEncoder, CAE, CHAE, TBAE, CTBAE, DACTBAE, DACHTBAE, NewAE
from shillml.models.nsdes import AutoEncoderDiffusion, AutoEncoderDiffusion2, AutoEncoderDrift
from shillml.models.nsdes import ambient_quadratic_variation_drift


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
    normal_penalty = torch.mean(torch.linalg.vector_norm(normal_projected_tangent_vector, ord=2, dim=1) ** 2)
    return normal_penalty


class MatrixMSELoss(nn.Module):
    def __init__(self, norm="fro", *args, **kwargs):
        """

        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, input_data: Tensor, target: Tensor) -> Tensor:
        """

        :param input_data:
        :param target:
        :return:
        """
        return torch.mean(torch.linalg.matrix_norm(input_data - target, ord=self.norm) ** 2)


class AELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, ae: AutoEncoder, x: Tensor, targets: Optional[Tensor] = None):
        """

        :param ae:
        :param x:
        :param targets:
        :return:
        """
        x_hat = ae.forward(x)
        return self.reconstruction_loss(x_hat, x)


class CAELoss(AELoss):
    def __init__(self, contractive_weight=0., *args, **kwargs):
        """

        :param contractive_weight:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.contractive_weight = contractive_weight
        self.contractive_reg = contractive_regularization

    def forward(self, cae: CAE, x: Tensor, targets: Optional[Tensor] = None):
        """

        :param cae:
        :param x:
        :param targets:
        :return:
        """
        x_hat, dpi = cae.forward(x)
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        contractive_penalty = self.contractive_weight * self.contractive_reg(dpi)
        total_loss = reconstruction_loss + contractive_penalty
        return total_loss


class CHAELoss(CAELoss):
    """

    """

    def __init__(self, contractive_weight=1., hessian_weight=1., *args, **kwargs):
        """

        :param contractive_weight:
        :param hessian_weight:
        :param args:
        :param kwargs:
        """
        super().__init__(contractive_weight, *args, **kwargs)
        self.hessian_weight = hessian_weight
        self.hessian_reg = hessian_regularization

    def forward(self, chae: CHAE, x: Tensor, targets: Optional[Tensor] = None):
        """

        :param chae:
        :param x:
        :param targets:
        :return:
        """
        x_hat, dpi, hessian_pi = chae.forward(x)
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        contractive_loss = self.contractive_weight * self.contractive_reg(dpi)
        hessian_loss = self.hessian_weight * self.hessian_reg(hessian_pi)
        total_loss = reconstruction_loss + contractive_loss + hessian_loss
        return total_loss


class TBAELoss(AELoss):
    """

    """

    def __init__(self,
                 tangent_bundle_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        """

        :param tangent_bundle_weight:
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.tangent_bundle_weight = tangent_bundle_weight
        self.tangent_bundle_reg = MatrixMSELoss(norm=norm)

    def forward(self, tbae: TBAE, x: Tensor, target: Optional[Tensor] = None):
        """

        :param tbae:
        :param x:
        :param target:
        :return:
        """
        x_hat, model_projection = tbae.forward(x)
        reconstruction_mse = self.reconstruction_loss(x_hat, x)
        tangent_mse = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, target)
        total_loss = reconstruction_mse + tangent_mse
        return total_loss


class CTBAELoss(TBAELoss, CAELoss):
    """

    """

    def __init__(self,
                 tangent_bundle_weight=1.,
                 contractive_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        """

        :param tangent_bundle_weight:
        :param contractive_weight:
        :param norm:
        :param args:
        :param kwargs:
        """
        CAELoss.__init__(self, contractive_weight=contractive_weight, *args, **kwargs)
        TBAELoss.__init__(self, tangent_bundle_weight=tangent_bundle_weight, norm=norm, *args, **kwargs)

    def forward(self, ctbae: CTBAE, x: Tensor, orthog_proj: Optional[Tensor] = None):
        """

        :param ctbae:
        :param x:
        :param orthog_proj:
        :return:
        """
        x_hat, dpi, model_projection = ctbae.forward(x)
        mse = self.reconstruction_loss(x_hat, x)
        contractive_penalty = self.contractive_reg(dpi) * self.contractive_weight
        tangent_error = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, orthog_proj)
        total_loss = mse + contractive_penalty + tangent_error
        return total_loss


class DACTBAELoss(CTBAELoss):
    """
    A custom loss function for the DACTBAE (Drift-Aligned Contractive Tangent Bundle Autoencoder) model,
    which extends the CTBAELoss by adding a term that penalizes the deviation of the drift vector field projected
    onto the normal space.

    This loss function includes multiple components:
    1. Reconstruction loss: Measures how well the reconstructed input matches the original input.
    2. Contractive regularization: Penalizes the Jacobian of the encoder to encourage smoother mappings.
    3. Tangent bundle regularization: Penalizes the difference between the model's projection onto the tangent bundle
       and the observed tangent bundle projection.
    4. Tangent drift loss: Penalizes the deviation of the drift vector projected onto the normal space.

    Attributes:
        tangent_drift_weight (float): Weight for the tangent drift loss component.
        tangent_drift_loss (callable): The function to compute the tangent drift loss.
    """

    def __init__(self,
                 contractive_weight=1.,
                 tangent_drift_weight=1.,
                 tangent_bundle_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        """

        :param contractive_weight:
        :param tangent_drift_weight:
        :param tangent_bundle_weight:
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(tangent_bundle_weight=tangent_bundle_weight,
                         contractive_weight=contractive_weight,
                         norm=norm, *args, **kwargs)
        self.tangent_drift_weight = tangent_drift_weight
        self.tangent_drift_loss = tangent_drift_loss

    def forward(self, dactbae: DACTBAE, x: Tensor, targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None):
        """

        :param dactbae:
        :param x:
        :param targets:
        :return:
        """
        x_hat, dpi, model_projection, decoder_hessian = dactbae.forward(x)
        observed_projection, observed_normal_projection, drift, cov = targets
        bbt_proxy = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
        qv = ambient_quadratic_variation_drift(bbt_proxy, decoder_hessian)
        tangent_vector = drift - 0.5 * qv
        normal_proj_vector = torch.bmm(observed_normal_projection, tangent_vector.unsqueeze(2))
        mse = self.reconstruction_loss(x_hat, x)
        contractive_penalty = self.contractive_reg(dpi) * self.contractive_weight
        tangent_bundle_error = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection,
                                                                                    observed_projection)
        tangent_drift_error = self.tangent_drift_weight * self.tangent_drift_loss(normal_proj_vector)
        total_loss = mse + contractive_penalty + tangent_bundle_error + tangent_drift_error
        return total_loss


class NewAELoss(CTBAELoss):
    """
    A custom loss function for an auto encoder with the following regularizations:
    1. Contractive--minimizing the Frobenius norm of the encoder's Jacobian
    2. Tangent Bundle regularization: minimizing the Frobenius error of the decoder's orthogonal projection against
    and observed one
    3. Ambient tangent drift alignment regularization: minimizes the square norm of the normal projected ambient
    tangent drift mu-0.5 q, where $q$ is approximated with the observed covariance
    4. Diffeomorphism regularization: the square Frobenius error of Dpi - g^{-1} Dphi^T

    Attributes:
        tangent_drift_weight (float): Weight for the tangent drift loss component.
        tangent_drift_loss (callable): The function to compute the tangent drift loss.
    """

    def __init__(self,
                 contractive_weight=1.,
                 tangent_drift_weight=1.,
                 tangent_bundle_weight=1.,
                 diffeo_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        """

        :param contractive_weight:
        :param tangent_drift_weight:
        :param tangent_bundle_weight:
        :param diffeomorphism_weight:
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(tangent_bundle_weight=tangent_bundle_weight,
                         contractive_weight=contractive_weight,
                         norm=norm, *args, **kwargs)
        self.tangent_drift_weight = tangent_drift_weight
        self.tangent_drift_loss = tangent_drift_loss
        self.diffeo_weight = diffeo_weight

    def forward(self, dactbae: NewAE, x: Tensor, targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None):
        """

        :param dactbae:
        :param x:
        :param targets:
        :return:
        """
        d = dactbae.intrinsic_dim
        # Get model outputs and regularization inputs
        x_hat, dpi, dphi, decoder_hessian = dactbae.forward(x)
        # Extract targets
        observed_projection, observed_normal_projection, drift, cov, orthonormal_frame = targets
        # Compute g = Dphi^T Dphi
        g = torch.bmm(dphi.mT, dphi)
        # Compute g^-1 (you can use torch.linalg.inv for this)
        g_inv = torch.linalg.inv(g)
        model_projection = torch.bmm(torch.bmm(dphi, g_inv), dphi.mT)
        bbt_proxy = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
        qv = ambient_quadratic_variation_drift(bbt_proxy, decoder_hessian)
        tangent_vector = drift - 0.5 * qv
        # This is slow
        # normal_proj_vector = torch.bmm(observed_normal_projection, tangent_vector.unsqueeze(2))
        # This is faster
        orthonormal_frame_times_td = torch.bmm(orthonormal_frame.mT, tangent_vector.unsqueeze(2))
        normal_proj_vector = tangent_vector.unsqueeze(2) - torch.bmm(orthonormal_frame, orthonormal_frame_times_td)
        normal_proj_vector = normal_proj_vector.squeeze(2)
        # Losses
        mse = self.reconstruction_loss(x_hat, x)
        contractive_penalty = self.contractive_weight * self.contractive_reg(dpi)
        tangent_bundle_error = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection,
                                                                                    observed_projection)
        tangent_drift_error = self.tangent_drift_weight * self.tangent_drift_loss(normal_proj_vector)
        # Compute the Frobenius norm term: ||Dpi - g^-1 Dphi^T||_F^2
        frobenius_term = self.diffeo_weight * torch.mean(torch.norm(torch.bmm(g, dpi) - dphi.mT, p='fro') ** 2)
        total_loss = mse + contractive_penalty + tangent_bundle_error + tangent_drift_error + frobenius_term
        return total_loss


class DACHTBAELoss(CHAELoss, DACTBAELoss):
    """
        A combined loss function class for DACHTBAE models, inheriting from both `CHAELoss` and `DACTBAELoss`.
        This loss function integrates multiple regularization terms including:
        - Contractive regularization
        - Tangent Bundle regularization
        - Drift alignment regularization
        - Hessian regularization

        The total loss is a weighted sum of these regularization terms along with a standard reconstruction loss.

        Parameters:
        -----------
        contractive_weight : float, optional (default=1.0)
            Weight applied to the contractive regularization term.
        hessian_weight : float, optional (default=1.0)
            Weight applied to the Hessian regularization term.
        tangent_bundle_weight : float, optional (default=1.0)
            Weight applied to the tangent bundle regularization term.
        tangent_drift_weight : float, optional (default=1.0)
            Weight applied to the tangent drift regularization term.
        norm : str, optional (default="fro")
            Norm type to be used in the regularization calculations.
        args : tuple, optional
            Additional positional arguments to be passed to the parent classes.
        kwargs : dict, optional
            Additional keyword arguments to be passed to the parent classes.

        Attributes:
        -----------
        contractive_weight : float
            Weight for the contractive regularization.
        hessian_weight : float
            Weight for the Hessian regularization.
        tangent_bundle_weight : float
            Weight for the tangent bundle regularization.
        tangent_drift_weight : float
            Weight for the tangent drift regularization.
    """

    def __init__(self,
                 contractive_weight=1.,
                 hessian_weight=1.,
                 tangent_bundle_weight=1.,
                 tangent_drift_weight=1.,
                 norm="fro",
                 *args,
                 **kwargs):
        """

        :param contractive_weight:
        :param hessian_weight:
        :param tangent_bundle_weight:
        :param tangent_drift_weight:
        :param norm:
        :param args:
        :param kwargs:
        """
        DACTBAELoss.__init__(self, tangent_drift_weight=tangent_drift_weight,
                             tangent_bundle_weight=tangent_bundle_weight,
                             contractive_weight=contractive_weight,
                             norm=norm, *args, **kwargs)
        CHAELoss.__init__(self, contractive_weight=contractive_weight,
                          hessian_weight=hessian_weight, *args, **kwargs)

    def forward(self, dachtbae: DACHTBAE, x: Tensor, targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None):
        """
        Computes the total loss for the DACHTBAE model, combining reconstruction loss with several regularization terms:
        - Contractive regularization
        - Tangent Bundle regularization
        - Drift alignment regularization
        - Hessian regularization

        The loss function is designed for autoencoders that aim to preserve geometric structures during encoding, such as
        those operating on manifolds.

        Parameters:
        -----------
        dachtbae : DACHTBAE
            The DACHTBAE model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : Optional[Union[Tensor, Tuple[Tensor, ...]]], optional
            Targets for the loss function, must include (observed_projection, observed_normal_projection, drift, cov).
            - observed_projection : Tensor
                The orthogonal projection of the input to the tangent space.
            - observed_normal_projection : Tensor
                The complement of the tangent projection, representing the normal space.
            - drift : Tensor
                The drift vector for each data point.
            - cov : Tensor
                The covariance matrix for each data point.

        Returns:
        --------
        total_loss : Tensor
            The computed total loss, which includes reconstruction loss and weighted regularization terms.
        """
        x_hat, dpi, model_projection, decoder_hessian, encoder_hessian = dachtbae.forward(x)
        observed_projection, observed_normal_projection, drift, cov = targets
        bbt_proxy = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
        qv = ambient_quadratic_variation_drift(bbt_proxy, decoder_hessian)
        tangent_vector = drift - 0.5 * qv
        normal_proj_vector = torch.bmm(observed_normal_projection, tangent_vector.unsqueeze(2))
        mse = self.reconstruction_loss(x_hat, x)
        contr = self.contractive_reg(dpi) * self.contractive_weight
        hess = self.hessian_weight * self.hessian_reg(encoder_hessian)
        tang = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, observed_projection)
        curv = self.tangent_drift_weight * self.tangent_drift_loss(normal_proj_vector)
        total_loss = mse + contr + tang + curv + hess
        return total_loss


class DiffusionLoss(nn.Module):
    """
        A custom loss function for training an AutoEncoderDiffusion model. This loss combines several components:
        - Mean squared error (MSE) between the predicted covariance and the target covariance.
        - MSE between the local covariance of the encoded space and the target encoded covariance.
        - A tangent drift loss that penalizes deviations in the normal projection of the drift vector.

        The total loss is a weighted sum of these components, allowing for control over the influence of the tangent drift loss.

        Parameters:
        -----------
        tangent_drift_weight : float, optional (default=0.0)
            Weight applied to the tangent drift loss component.
        norm : str, optional (default="fro")
            Norm type to be used in the MSE loss calculations. The norm is applied when computing the matrix MSE.

        Attributes:
        -----------
        tangent_drift_weight : float
            Weight applied to the tangent drift loss component.
        cov_mse : MatrixMSELoss
            MSE loss instance for comparing model and target covariances in the ambient space.
        local_cov_mse : MatrixMSELoss
            MSE loss instance for comparing local covariances in the encoded space.
        tangent_drift_loss : function
            Function used to calculate the tangent drift loss component.
    """

    def __init__(self, tangent_drift_weight=0.0, norm="fro"):
        """

        :param tangent_drift_weight: weight for the tangent drift alignment penalty
        :param norm: matrix norm for the covariance error
        """
        super().__init__()
        self.tangent_drift_weight = tangent_drift_weight
        self.cov_mse = MatrixMSELoss(norm=norm)
        self.local_cov_mse = MatrixMSELoss(norm=norm)
        self.tangent_drift_loss = tangent_drift_loss

    def forward(self, ae_diffusion: AutoEncoderDiffusion, x, targets):
        """
        Computes the total loss for the AutoEncoderDiffusion model.

        Parameters:
        -----------
        ae_diffusion : AutoEncoderDiffusion
            The AutoEncoderDiffusion model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the target ambient drift vector, target ambient covariance matrix,
            and target encoded covariance matrix (mu, cov, encoded_cov).

        Returns:
        --------
        total_loss : Tensor
            The computed total loss combining covariance MSE, local covariance MSE, and weighted tangent drift loss.
        """
        model_cov, normal_proj, qv, bbt = ae_diffusion.forward(x)
        ambient_drift, ambient_cov, encoded_cov = targets
        tangent_vector = ambient_drift - 0.5 * qv
        normal_proj_vector = torch.bmm(normal_proj, tangent_vector.unsqueeze(2))
        cov_mse = self.cov_mse(model_cov, ambient_cov)
        # Add local cov mse
        local_cov_mse = self.local_cov_mse(bbt, encoded_cov)
        normal_bundle_loss = self.tangent_drift_loss(normal_proj_vector)
        total_loss = 0.5 * cov_mse + 0.5 * local_cov_mse + self.tangent_drift_weight * normal_bundle_loss
        return total_loss


class DiffusionLoss2(nn.Module):
    """
        A custom loss function for training an AutoEncoderDiffusion model. This loss combines several components:
        - Mean squared error (MSE) between the predicted covariance and the target covariance.
        - MSE between the local covariance of the encoded space and the target encoded covariance.
        - A tangent drift loss that penalizes deviations in the normal projection of the drift vector.

        The total loss is a weighted sum of these components, allowing for control over the influence of the tangent drift loss.

        Parameters:
        -----------
        tangent_drift_weight : float, optional (default=0.0)
            Weight applied to the tangent drift loss component.
        norm : str, optional (default="fro")
            Norm type to be used in the MSE loss calculations. The norm is applied when computing the matrix MSE.

        Attributes:
        -----------
        tangent_drift_weight : float
            Weight applied to the tangent drift loss component.
        cov_mse : MatrixMSELoss
            MSE loss instance for comparing model and target covariances in the ambient space.
        local_cov_mse : MatrixMSELoss
            MSE loss instance for comparing local covariances in the encoded space.
        tangent_drift_loss : function
            Function used to calculate the tangent drift loss component.
    """

    def __init__(self, tangent_drift_weight=0.0, norm="fro"):
        """

        :param tangent_drift_weight: weight for the tangent drift alignment penalty
        :param norm: matrix norm for the covariance error
        """
        super().__init__()
        self.tangent_drift_weight = tangent_drift_weight
        self.cov_mse = MatrixMSELoss(norm=norm)
        self.local_cov_mse = MatrixMSELoss(norm=norm)
        self.tangent_drift_loss = tangent_drift_loss

    def forward(self, ae_diffusion: AutoEncoderDiffusion2, x, targets):
        """
        Computes the total loss for the AutoEncoderDiffusion model.

        Parameters:
        -----------
        ae_diffusion : AutoEncoderDiffusion
            The AutoEncoderDiffusion model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the target ambient drift vector, target ambient covariance matrix,
            and target encoded covariance matrix (mu, cov, encoded_cov).

        Returns:
        --------
        total_loss : Tensor
            The computed total loss combining covariance MSE, local covariance MSE, and weighted tangent drift loss.
        """
        dphi, g, qv, bbt = ae_diffusion.forward(x)
        ambient_drift, ambient_cov, encoded_cov, orthonormal_frame = targets
        tangent_vector = ambient_drift - 0.5 * qv
        orthonormal_frame_times_td = torch.bmm(orthonormal_frame.mT, tangent_vector.unsqueeze(2))
        normal_proj_vector = tangent_vector.unsqueeze(2) - torch.bmm(orthonormal_frame, orthonormal_frame_times_td)
        normal_proj_vector = normal_proj_vector.squeeze(2)

        # cov_mse = self.cov_mse(model_cov, ambient_cov)
        # Add local cov mse
        local_cov_mse = self.local_cov_mse(bbt, encoded_cov)
        normal_bundle_loss = self.tangent_drift_loss(normal_proj_vector)
        # New loss term:
        # Compute Dphi^T Sigma Dphi
        transformed_cov = torch.bmm(dphi.mT, torch.bmm(ambient_cov, dphi))
        # Now compute g_inv Dphi^T Sigma Dphi g_inv
        transformed_latent_cov = torch.bmm(g, torch.bmm(bbt, g))
        # Frobenius norm term: ||X - g^-1 Dphi^T Sigma Dphi g^-1||_F^2
        # where X is the local model covariance bbt or
        # ||gXg - Dphi^T Sigma Dphi||_F^2
        frobenius_term = torch.mean(torch.norm(transformed_latent_cov - transformed_cov, p='fro') ** 2)
        total_loss = local_cov_mse + self.tangent_drift_weight * normal_bundle_loss + 0.1 * frobenius_term
        return total_loss


class DiffusionLoss3(nn.Module):
    """
        A custom loss function for training an AutoEncoderDiffusion model. This loss combines several components:
        - Mean squared error (MSE) between the predicted covariance and the target covariance.
        - A tangent drift loss that penalizes deviations in the normal projection of the drift vector.

        The total loss is a weighted sum of these components, allowing for control over the influence of the tangent drift loss.

        Parameters:
        -----------
        tangent_drift_weight : float, optional (default=0.0)
            Weight applied to the tangent drift loss component.
        norm : str, optional (default="fro")
            Norm type to be used in the MSE loss calculations. The norm is applied when computing the matrix MSE.

        Attributes:
        -----------
        tangent_drift_weight : float
            Weight applied to the tangent drift loss component.
        cov_mse : MatrixMSELoss
            MSE loss instance for comparing model and target covariances in the ambient space.
        tangent_drift_loss : function
            Function used to calculate the tangent drift loss component.
    """

    def __init__(self, latent_cov_weight=0.5, ambient_cov_weight=0.5, tangent_drift_weight=0.0, norm="fro",
                 normalize=False):
        """

        :param tangent_drift_weight: weight for the tangent drift alignment penalty
        :param norm: matrix norm for the covariance error
        """
        super().__init__()
        self.tangent_drift_weight = tangent_drift_weight
        self.cov_mse = MatrixMSELoss(norm=norm)
        self.tangent_drift_loss = tangent_drift_loss
        self.latent_cov_weight = latent_cov_weight
        self.ambient_cov_weight = ambient_cov_weight
        self.normalize = normalize

    def forward(self, ae_diffusion: AutoEncoderDiffusion2, x, targets):
        """
        Computes the total loss for the AutoEncoderDiffusion model.

        Parameters:
        -----------
        ae_diffusion : AutoEncoderDiffusion
            The AutoEncoderDiffusion model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the target ambient drift vector, target ambient covariance matrix,
            and target encoded covariance matrix (mu, cov, encoded_cov, H, N).

        Returns:
        --------
        total_loss : Tensor
            The computed total loss combining covariance MSE, local covariance MSE, and weighted tangent drift loss.
        """
        dphi, g, qv, bbt = ae_diffusion.forward(x)
        ambient_drift, ambient_cov, encoded_cov, orthonormal_frame = targets

        tangent_vector = ambient_drift - 0.5 * qv
        if self.normalize:
            tangent_vector = tangent_vector / torch.linalg.vector_norm(tangent_vector, ord=2, dim=1, keepdim=True)

        htv = torch.bmm(orthonormal_frame.mT, tangent_vector.unsqueeze(2))
        hhtv = torch.bmm(orthonormal_frame, htv).squeeze(2)
        # This is equivalent to N(mu-0.5 q). It is just v-HH^T v where v = mu-0.5 q.
        normal_proj_vector = tangent_vector - hhtv
        tangent_drift_penalty = torch.mean(torch.linalg.vector_norm(normal_proj_vector, ord=2, dim=1) ** 2)
        model_cov = torch.bmm(dphi, bbt)
        model_cov = torch.bmm(model_cov, dphi.mT)
        cov_mse = self.cov_mse(model_cov, ambient_cov)
        local_cov_mse = self.cov_mse(bbt, encoded_cov)
        total_loss = self.ambient_cov_weight * cov_mse
        total_loss += self.latent_cov_weight * local_cov_mse
        total_loss += self.tangent_drift_weight * tangent_drift_penalty
        return total_loss


class LatentDiffusionLoss(nn.Module):
    def __init__(self, tangent_drift_weight=0., norm="fro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tangent_drift_weight = tangent_drift_weight
        self.cov_mse = MatrixMSELoss(norm=norm)
        self.tangent_drift_loss = tangent_drift_loss

    def forward(self, ae_diffusion: AutoEncoderDiffusion2, x, targets):
        dphi, g, qv, bbt = ae_diffusion.forward(x)
        ambient_drift, ambient_cov, encoded_cov, orthonormal_frame = targets
        tangent_vector = ambient_drift - 0.5 * qv
        htv = torch.bmm(orthonormal_frame.mT, tangent_vector.unsqueeze(2))
        hhtv = torch.bmm(orthonormal_frame, htv).squeeze(2)
        normal_proj_vector = tangent_vector - hhtv
        tangent_drift_penalty = torch.mean(torch.linalg.vector_norm(normal_proj_vector, ord=2, dim=1) ** 2)
        model_cov = torch.bmm(dphi, bbt)
        model_cov = torch.bmm(model_cov, dphi.mT)
        cov_mse = self.cov_mse(model_cov, ambient_cov)
        total_loss = cov_mse + self.tangent_drift_weight * tangent_drift_penalty
        return total_loss


class DriftMSELoss(nn.Module):
    """
    A custom loss function for training an AutoEncoderDrift model. This loss function combines:
    - Mean squared error (MSE) between the model-predicted ambient drift and the observed ambient drift.
    - MSE between the model-predicted latent drift and the true latent drift.

    The loss is computed by first transforming the observed ambient drift into the latent space using the model's
    decoder jacobian and neural metric tensor. This allows for a direct comparison between model predictions and targets
    in both the ambient and latent spaces.

    Parameters:
    -----------
    None. Inherits from nn.Module.

    Methods:
    --------
    forward(drift_model: AutoEncoderDrift, x: Tensor, targets: Tuple[Tensor, Tensor]) -> Tensor
        Computes the loss given the input data, the drift model, and the target drift and covariance.
    """

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(drift_model: AutoEncoderDrift, x: Tensor,
                targets: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Computes the loss for the AutoEncoderDrift model.

        Parameters:
        -----------
        drift_model : AutoEncoderDrift
            The AutoEncoderDrift model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the observed ambient drift vector and the target encoded covariance matrix
            (mu, encoded_cov).

        Returns:
        --------
        loss : Tensor
            The computed loss combining ambient drift error and latent drift error.
        """
        observed_ambient_drift, encoded_cov = targets
        z = drift_model.autoencoder.encoder(x)
        latent_drift = drift_model.latent_sde.drift_net(z)
        decoder_hessian = drift_model.autoencoder.decoder_hessian(z)
        q = ambient_quadratic_variation_drift(encoded_cov, decoder_hessian)
        model_ambient_drift = drift_model(x)
        dphi = drift_model.autoencoder.decoder_jacobian(z)
        ginv = torch.linalg.inv(drift_model.autoencoder.neural_metric_tensor(z))
        vec = observed_ambient_drift - 0.5 * q
        true_latent_drift = torch.bmm(ginv, torch.bmm(dphi.mT, vec.unsqueeze(2))).squeeze()
        ambient_error = torch.mean(
            torch.linalg.vector_norm(model_ambient_drift - observed_ambient_drift, ord=2, dim=1) ** 2)
        latent_error = torch.mean(torch.linalg.vector_norm(latent_drift - true_latent_drift, ord=2, dim=1) ** 2)
        return ambient_error + latent_error


class DriftMSELoss2(nn.Module):
    """
    A custom loss function for training an AutoEncoderDrift model. This loss function combines:
    - Mean squared error (MSE) between the model-predicted local drift and the observed local drift target.
    - MSE between the model-predicted latent drift and the true latent drift.

    The loss is computed by first transforming the observed ambient drift into the latent space using the model's
    decoder jacobian and neural metric tensor. This allows for a direct comparison between model predictions and targets
    in both the ambient and latent spaces.

    Parameters:
    -----------
    None. Inherits from nn.Module.

    Methods:
    --------
    forward(drift_model: AutoEncoderDrift, x: Tensor, targets: Tuple[Tensor, Tensor]) -> Tensor
        Computes the loss given the input data, the drift model, and the target drift and covariance.
    """

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(drift_model: AutoEncoderDrift, x: Tensor,
                targets: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Computes the loss for the AutoEncoderDrift model.

        Parameters:
        -----------
        drift_model : AutoEncoderDrift
            The AutoEncoderDrift model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the observed ambient drift vector and the target encoded covariance matrix
            (mu, encoded_cov).

        Returns:
        --------
        loss : Tensor
            The computed loss combining ambient drift error and latent drift error.
        """
        observed_ambient_drift, encoded_cov = targets
        z = drift_model.autoencoder.encoder(x)
        latent_drift = drift_model.latent_sde.drift_net(z)
        decoder_hessian = drift_model.autoencoder.decoder_hessian(z)
        q = ambient_quadratic_variation_drift(encoded_cov, decoder_hessian)
        dphi = drift_model.autoencoder.decoder_jacobian(z)
        g = drift_model.autoencoder.neural_metric_tensor(z)
        tangent_drift = observed_ambient_drift - 0.5 * q
        true_latent_drift = torch.bmm(dphi.mT, tangent_drift.unsqueeze(2)).squeeze()
        g_latent_drift = torch.bmm(g, latent_drift.unsqueeze(2)).squeeze(2)
        latent_error = torch.mean(torch.linalg.vector_norm(g_latent_drift - true_latent_drift, ord=2, dim=1) ** 2)
        return latent_error


class DriftMSELoss3(nn.Module):
    """
    A custom loss function for training an AutoEncoderDrift model. This loss function computes:
    - Mean squared error (MSE) between the model-predicted ambient drift and the observed ambient drift.

    Parameters:
    -----------
    None. Inherits from nn.Module.

    Methods:
    --------
    forward(drift_model: AutoEncoderDrift, x: Tensor, targets: Tuple[Tensor, Tensor]) -> Tensor
        Computes the loss given the input data, the drift model, and the target drift and covariance.
    """

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(drift_model: AutoEncoderDrift, x: Tensor,
                targets: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Computes the loss for the AutoEncoderDrift model.

        Parameters:
        -----------
        drift_model : AutoEncoderDrift
            The AutoEncoderDrift model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the observed ambient drift vector and the target encoded covariance matrix
            (mu, encoded_cov).

        Returns:
        --------
        loss : Tensor
            The computed loss combining ambient drift error and latent drift error.
        """
        observed_ambient_drift, encoded_cov = targets
        model_ambient_drift = drift_model(x)
        ambient_sq_error = torch.linalg.vector_norm(model_ambient_drift - observed_ambient_drift, ord=2, dim=1) ** 2
        ambient_drift_mse = torch.mean(ambient_sq_error)
        return ambient_drift_mse


class LatentDriftMSE(nn.Module):
    """
    A custom loss function for training an AutoEncoderDrift model. This loss function computes:
    - Mean squared error (MSE) between the model-predicted latent drift and the observed latent drift, encoded

    Parameters:
    -----------
    None. Inherits from nn.Module.

    Methods:
    --------
    forward(drift_model: AutoEncoderDrift, x: Tensor, targets: Tuple[Tensor, Tensor]) -> Tensor
        Computes the loss given the input data, the drift model, and the target drift and covariance.
    """

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(drift_model: AutoEncoderDrift, x: Tensor,
                targets: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Computes the loss for the AutoEncoderDrift model.

        Parameters:
        -----------
        drift_model : AutoEncoderDrift
            The AutoEncoderDrift model whose output is being evaluated.
        x : Tensor
            Input tensor to the autoencoder.
        targets : tuple of Tensors
            A tuple containing the observed ambient drift vector and the target encoded covariance matrix
            (mu, encoded_cov).

        Returns:
        --------
        loss : Tensor
            The computed loss combining ambient drift error and latent drift error.
        """
        observed_ambient_drift, _ = targets
        z, _, _, q = drift_model.compute_sde_manifold_tensors(x)
        model_latent_drift = drift_model.latent_sde.drift_net.forward(z)
        dpi = drift_model.autoencoder.encoder_jacobian(x)
        tangent_drift_vector = observed_ambient_drift - 0.5 * q
        observed_latent_drift = torch.bmm(dpi, tangent_drift_vector.unsqueeze(2)).squeeze(2)
        latent_sq_error = torch.linalg.vector_norm(model_latent_drift - observed_latent_drift, ord=2, dim=1) ** 2
        latent_drift_mse = torch.mean(latent_sq_error)
        return latent_drift_mse
