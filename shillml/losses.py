import torch
import torch.nn as nn
from torch import Tensor
from typing import Any


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
    def __init__(self, tangent_bundle_weight=1., observed_projection=None, norm="fro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tangent_bundle_weight = tangent_bundle_weight
        self.mse_loss = nn.MSELoss()
        self.tangent_bundle_reg = TangentBundleLoss(norm=norm)
        self.observed_projection = observed_projection

    def forward(self, output, x):
        x_hat, model_projection = output
        reconstruction_mse = self.mse_loss(x_hat, x)
        tangent_mse = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection)
        total_loss = reconstruction_mse + tangent_mse
        return total_loss


class CTBAELoss(nn.Module):
    def __init__(self, contractive_weight=1., tangent_bundle_weight=1., observed_projection=None, norm="fro", *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.contractive_weight = contractive_weight
        self.tangent_bundle_weight = tangent_bundle_weight
        self.mse_loss = nn.MSELoss()
        self.tangent_bundle_reg = TangentBundleLoss(norm=norm)
        self.contractive_reg = ContractiveRegularization()
        self.observed_projection = observed_projection

    def forward(self, output, x):
        x_hat, dpi, model_projection = output
        mse = self.mse_loss(x_hat, x)
        contr = self.contractive_reg(dpi) * self.contractive_weight
        tang = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection)
        total_loss = mse + contr + tang
        return total_loss


class CurvatureCTBAELoss(nn.Module):
    def __init__(self, contractive_weight=1., tangent_bundle_weight=1., curvature_weight=1., observed_projection=None,
                 norm="fro", *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.contractive_weight = contractive_weight
        self.tangent_bundle_weight = tangent_bundle_weight
        self.curvature_weight = curvature_weight
        self.mse_loss = nn.MSELoss()
        self.tangent_bundle_reg = TangentBundleLoss(norm=norm)
        self.contractive_reg = ContractiveRegularization()
        self.curvature_reg = NormalBundleLoss()
        self.observed_projection = observed_projection
        n, D, _ = self.observed_projection.size()
        self.extrinsic_dim = D
        self.observed_normal_proj = torch.eye(D).expand(n, D, D) - self.observed_projection

    def forward(self, output, targets):
        x_hat, dpi, model_projection, decoder_hessian = output
        x, ambient_drift, ambient_cov = targets
        bbt_proxy = torch.bmm(torch.bmm(dpi, ambient_cov), dpi.mT)
        qv = torch.stack(
            [torch.einsum("nii -> n", torch.bmm(bbt_proxy, decoder_hessian[:, i, :, :])) for i in
             range(self.extrinsic_dim)])
        qv = qv.T
        tangent_vector = ambient_drift - 0.5 * qv
        normal_proj_vector = torch.bmm(self.observed_normal_proj, tangent_vector.unsqueeze(2))
        mse = self.mse_loss(x_hat, x)
        contr = self.contractive_reg(dpi) * self.contractive_weight
        tang = self.tangent_bundle_weight * self.tangent_bundle_reg(model_projection, self.observed_projection)
        curv = self.curvature_weight * self.curvature_reg(normal_proj_vector)
        total_loss = mse + contr + tang + curv
        return total_loss


class CovarianceMSELoss(nn.Module):
    def __init__(self, norm="fro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, model_cov: Tensor, observed_cov: Tensor) -> Tensor:
        """
        Computes the mean squared error (MSE) between observed and model covariance matrices.

        Args:
            model_cov (Tensor): Model covariance matrix.
            observed_cov (Tensor): Observed covariance matrix.

        Returns:
            Tensor: Mean squared error between observed and model covariance matrices.
        """
        cov_error = torch.linalg.matrix_norm(model_cov - observed_cov, ord=self.norm)
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
    def __init__(self, normal_bundle_weight=0.0, norm="fro", normalize: bool = True):
        super().__init__()
        self.normal_bundle_weight = normal_bundle_weight
        self.cov_mse = CovarianceMSELoss(norm=norm)
        self.local_cov_mse = CovarianceMSELoss(norm=norm)
        self.normal_bundle_loss = NormalBundleLoss()
        self.normalize = normalize

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
        if self.normalize:
            norms = torch.linalg.vector_norm(normal_proj_vector.squeeze(-1), dim=1, keepdim=True)
            normal_proj_vector = normal_proj_vector.squeeze(-1) / norms
            normal_proj_vector = normal_proj_vector.unsqueeze(2)

        cov_mse = self.cov_mse(model_cov, observed_cov)
        # Add local cov mse
        local_cov_mse = self.local_cov_mse(bbt, encoded_observed_cov)
        normal_bundle_loss = self.normal_bundle_loss(normal_proj_vector)
        total_loss = cov_mse + local_cov_mse + self.normal_bundle_weight * normal_bundle_loss
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