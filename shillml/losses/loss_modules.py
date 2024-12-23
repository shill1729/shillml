from torch import Tensor
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from shillml.models.autoencoders import AutoEncoder1
from shillml.models.nsdes import ambient_quadratic_variation_drift
from shillml.models.autoencoders.lae import pairwise_distances, computeW


# TODO this is a redundant penalty. Should we delete it or keep it to demonstrate that numerically?
class NormalComponentReconstructionLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(points, reconstructed_points, orthonormal_frame):
        r = reconstructed_points - points
        y = torch.bmm(orthonormal_frame, torch.bmm(orthonormal_frame.mT, r.unsqueeze(2))).squeeze(2)
        return torch.mean(torch.linalg.vector_norm(r - y, dim=1) ** 2)


class TangentSpaceAnglesLoss(nn.Module):
    """ Equivalent to minimizing the frobenius error of P-P_hat, we can make the angle between subspaces zero.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(observed_frame, decoder_jacobian, metric_tensor):
        _, _, d = decoder_jacobian.size()
        # TODO: More efficient inverses and square roots of matrices, or avoid them entirely
        ginv = torch.linalg.inv(metric_tensor)
        evals, evecs = torch.linalg.eigh(ginv)
        # Square root matrix via EVD:
        gframe = torch.bmm(evecs, torch.bmm(torch.diag_embed(torch.sqrt(evals)), evecs.mT))
        model_frame = torch.bmm(decoder_jacobian, gframe)
        a = torch.bmm(observed_frame.mT, model_frame)
        u, sigma, vt = torch.linalg.svd(a)
        return 2 * torch.mean(d - torch.sum(sigma ** 2, dim=1))


class MatrixMSELoss(nn.Module):
    """
        Compute the mean square error between two matrices under any matrix-norm.
    """

    def __init__(self, norm="fro", *args, **kwargs):
        """
            Compute the mean square error between two matrices under any matrix-norm.

        :param norm: the matrix norm to use: "fro", "nuc", -2, 2, inf, -inf, etc
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, input_data: Tensor, target: Tensor) -> Tensor:
        """
            Compute the mean square error between two matrices under any matrix-norm.
        :param input_data: tensor of shape (n, a, b)
        :param target: tensor of shape (n, a, b)
        :return: tensor of shape (1, ).
        """
        square_error = torch.linalg.matrix_norm(input_data - target, ord=self.norm) ** 2
        mse = torch.mean(square_error)
        return mse


class ReconstructionLoss(nn.Module):
    """
        Compute the reconstruction loss between an auto-encoder and a given point cloud.
    """

    def __init__(self, *args, **kwargs):
        """
            Compute the reconstruction loss between an auto-encoder and a given point cloud.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, reconstructed_points: Tensor, points: Tensor) -> Tensor:
        """

        :param reconstructed_points: tensor of shape (n, D)
        :param points: tensor of shape (n, D)
        :return:
        """
        reconstruction_error = self.reconstruction_loss(reconstructed_points, points)
        return reconstruction_error


class ContractiveRegularization(nn.Module):
    """
        Ensure an auto-encoder is contractive by bounding the Frobenius norm of its encoder's Jacobian.
    """

    def __init__(self, norm="fro", *args, **kwargs):
        """
            Ensure an auto-encoder is contractive by bounding the Frobenius norm of its encoder's Jacobian.
        :param norm: the matrix norm to use: "fro", "nuc", -2, 2, inf, -inf, etc
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, encoder_jacobian: Tensor) -> Tensor:
        """
            Ensure an auto-encoder is contractive by bounding the Frobenius norm of its encoder's Jacobian.

        :param encoder_jacobian: tensor of shape (n, d, D)
        :return: tensor of shape (1, ).
        """
        encoder_jacobian_norm = torch.linalg.matrix_norm(encoder_jacobian, ord=self.norm)
        contraction_penalty = torch.mean(encoder_jacobian_norm ** 2)
        return contraction_penalty


class DeficientRankPenalty(nn.Module):
    """
        Ensure the decoder's Jacobian has full rank by bounding the smallest singular value away from zero.
    """

    def __init__(self, scale=1., *args, **kwargs):
        """
            Ensure the decoder's Jacobian has full rank by bounding the smallest singular value away from zero.

        :param scale:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.scale = scale

    def forward(self, decoder_jacobian: Tensor) -> Tensor:
        """
            Ensure the decoder's Jacobian has full rank by bounding the smallest singular value away from zero.

        :param decoder_jacobian: tensor of shape (n, D, d)
        :return: tensor of shape (1, ).
        """
        min_sv = torch.linalg.matrix_norm(decoder_jacobian, ord=-2)
        return torch.mean(torch.exp(-self.scale * min_sv))
        # return torch.mean(torch.abs(min_sv - self.scale) ** 2)


class TangentBundleRegularization(nn.Module):
    """
        A regularization term to train the autoencoder's orthogonal projection to approximate an observed orthogonal
        projection
    """

    def __init__(self, norm="fro", *args, **kwargs):
        """
            A regularization term to train the autoencoder's orthogonal projection to approximate an observed
            orthogonal projection
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.matrix_mse = MatrixMSELoss(norm)

    def forward(self, decoder_jacobian: Tensor, metric_tensor: Tensor, true_projection: Tensor):
        """
            A regularization term to train the autoencoder's orthogonal projection to approximate an observed
            orthogonal projection

        :param decoder_jacobian: tensor of shape (n, D, d)
        :param metric_tensor: tensor of shape (n, d, d)
        :param true_projection: tensor of shape (n, D, D), the observed orthogonal projection onto the tangent space
        :return:
        """
        # TODO: Should we compute the inverse metric tensor inside this function or have it passed as it currently is?
        # TODO: Also more efficient inverses?
        inverse_metric_tensor = torch.linalg.inv(metric_tensor)
        model_projection = torch.bmm(decoder_jacobian, torch.bmm(inverse_metric_tensor, decoder_jacobian.mT))
        return self.matrix_mse(model_projection, true_projection)


class TangentDriftAlignment(nn.Module):
    """
        A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.
    """

    def __init__(self, *args, **kwargs):
        """
            A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()

    @staticmethod
    def forward(encoder_jacobian, decoder_hessian, ambient_cov, ambient_drift, observed_frame):
        """
            A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.

        :param encoder_jacobian: tensor of shape (n, d, D)
        :param decoder_hessian: tensor of shape (n, D, d)
        :param ambient_cov: tensor of shape (n, D, D)
        :param ambient_drift: tensor of shape (n, D)
        :param observed_frame: tensor of shape (n, D, d)
        :return: tensor of shape (1, ).
        """
        # Ito's lemma from D -> d gives a proxy to the local covariance using the Jacobian of the encoder
        bbt_proxy = torch.bmm(torch.bmm(encoder_jacobian, ambient_cov), encoder_jacobian.mT)
        # The QV correction from d -> D with the proxy-local cov: q^i = < bb^T, nabla^2 phi^i >_F
        qv = ambient_quadratic_variation_drift(bbt_proxy, decoder_hessian)
        # The ambient drift mu = Dphi a + 0.5 q should satisfy v := mu-0.5 q has P v = 0 since v = Dphi a is tangent
        # to the manifold
        tangent_drift = ambient_drift - 0.5 * qv
        # Compute v-Pv and minimize this norm. Use P=HH^T to avoid DxD products--Is this really necessary?
        frame_transpose_times_tangent_vector = torch.bmm(observed_frame.mT, tangent_drift.unsqueeze(2))
        tangent_projected = torch.bmm(observed_frame, frame_transpose_times_tangent_vector).squeeze(2)
        normal_projected_tangent_drift = tangent_drift - tangent_projected
        return torch.mean(torch.linalg.vector_norm(normal_projected_tangent_drift, ord=2) ** 2)


class TangentDriftAlignment2(nn.Module):
    """
        A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.
    """

    def __init__(self, normalize=False, *args, **kwargs):
        """
            A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reconstruction_loss = nn.MSELoss()
        self.normalize = normalize

    def forward(self, encoder_jacobian, decoder_hessian, ambient_cov, ambient_drift, observed_normal_proj):
        """
            A regularization term to align the higher-order geometry of an autoencoder with an observed
        ambient drift. Specifically, the ambient drift minus the 2nd-order ito-correction term should be
        tangent to the manifold. Hence the orthogonal projection onto the normal bundle should be zero. We minimize
        this norm to make it zero for an observed orthgonal projection. The model inputs go into the
        ito-correction term. A proxy to the local-covariance is used by encoding the ambient covariance via
        Ito's lemma when mapping from D -> d. The ito-correction term also uses the Hessian of the decoder,
        so this is a second-order regularization.

        :param encoder_jacobian: tensor of shape (n, d, D)
        :param decoder_hessian: tensor of shape (n, D, d)
        :param ambient_cov: tensor of shape (n, D, D)
        :param ambient_drift: tensor of shape (n, D)
        :param observed_normal_proj: tensor of shape (n, D, D)
        :return: tensor of shape (1, ).
        """
        # Ito's lemma from D -> d gives a proxy to the local covariance using the Jacobian of the encoder
        bbt_proxy = torch.bmm(torch.bmm(encoder_jacobian, ambient_cov), encoder_jacobian.mT)
        # The QV correction from d -> D with the proxy-local cov: q^i = < bb^T, nabla^2 phi^i >_F
        qv = ambient_quadratic_variation_drift(bbt_proxy, decoder_hessian)
        # The ambient drift mu = Dphi a + 0.5 q should satisfy v := mu-0.5 q has P v = 0 since v = Dphi a is tangent
        # to the manifold
        tangent_drift = ambient_drift - 0.5 * qv
        if self.normalize:
            tangent_drift = tangent_drift / torch.linalg.vector_norm(tangent_drift, ord=2, dim=1, keepdim=True)
        # Compute N . (mu-0.5 q)
        normal_projected_tangent_drift = torch.bmm(observed_normal_proj, tangent_drift.unsqueeze(2)).squeeze(2)
        # Return the mean of the norm squared
        loss = torch.mean(torch.linalg.vector_norm(normal_projected_tangent_drift, ord=2) ** 2)
        return loss


class DiffeomorphicRegularization1(nn.Module):
    """
        A naive method to ensure diffeomorphism conditions for an auto-encoder pair
    """

    def __init__(self, norm="fro", *args, **kwargs):
        """
            A naive method to ensure diffeomorphism conditions for an auto-encoder pair
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.matrix_mse = MatrixMSELoss(norm)

    def forward(self, decoder_jacobian: Tensor, encoder_jacobian: Tensor):
        """
            A naive method to ensure diffeomorphism conditions for an auto-encoder pair

        :param decoder_jacobian: tensor of shape (n, D, d)
        :param encoder_jacobian: tensor of shape (n, d, D)
        :return: tensor of shape (1, ).
        """
        # For large D >> d, it is more memory efficient to store d x d, therefore we compute Dpi Dphi
        model_product = torch.bmm(encoder_jacobian, decoder_jacobian)
        # Subtract identity matrix in-place without expanding across batches
        n, d, _ = model_product.size()
        # Create a diagonal matrix in-place
        diag_indices = torch.arange(d)
        model_product[:, diag_indices, diag_indices] -= 1.0
        # Compute the matrix MSE between (Dpi * Dphi) and I without explicitly creating I
        return self.matrix_mse(model_product, torch.zeros_like(model_product))


class VarianceLogVolume(nn.Module):
    """
        A method to ensure diffeomorphism conditions for an auto-encoder pair
    """

    def __init__(self, *args, **kwargs):
        """
            A method to ensure diffeomorphism conditions for an auto-encoder pair
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def forward(self, metric_tensor: Tensor):
        """
            A method to ensure diffeomorphism conditions for an auto-encoder pair

        :param metric_tensor: tensor of shape (n, d, d)
        :return:
        """
        # TODO: should we pass the metric tensor (current) or just compute it internally via g=Dphi^T Dphi?
        volume = 0.5 * torch.log(torch.linalg.det(metric_tensor))
        return torch.var(volume)


class OrthogonalCoordinatesRegularization(nn.Module):
    def __init__(self, *args, **kwargs):
        """
            A regularization to encourage orthogonal coordinates
        :param norm:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def forward(self, g):
        # Compute the Frobenius norm squared of the metric tensor g
        frobenius_norm_sq = torch.linalg.matrix_norm(g, ord="fro") ** 2

        # Compute the sum of squared off-diagonal elements
        # g_ij for i ≠ j means subtracting out the diagonal elements (which are included in g ** 2)
        diag_elements_sq = torch.sum(torch.diagonal(g, dim1=-2, dim2=-1) ** 2, dim=-1)

        # The penalty is the sum of off-diagonal squared elements
        off_diag_penalty = frobenius_norm_sq - diag_elements_sq

        # Regularization term
        regularization = torch.sum(off_diag_penalty)

        return regularization


# TODO add the kernel parameter to this and fit the X data and W and store it as attributes?
class LaplacianReg(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, affinity_matrix, latent_distances):
        # Element-wise multiplication between W and D
        weighted_distances = affinity_matrix * latent_distances
        # Sum all elements to get the final result
        loss = 0.5 * torch.sum(weighted_distances)
        return loss


# To add a new loss regularization, simply add a weight to the weight class, then modify TotalLoss
@dataclass
class LossWeights:
    reconstruction: float = 1.
    rank_penalty_weight: float = 0.0
    encoder_contraction_weight: float = 0.0
    decoder_contraction_weight: float = 0.0
    tangent_space_error_weight: float = 0.0
    tangent_angle_weight: float = 0.0
    tangent_drift_weight: float = 0.0
    diffeomorphism_reg1: float = 0.0
    variance_logdet: float = 0.0
    normal_component_recon: float = 0.0
    orthogonal_penalty_weight: float = 0.0
    laplacian_weight: float = 0.0


class TotalLoss(nn.Module):
    def __init__(self, weights: LossWeights, norm="fro", normalize=False, scale=1., affinity=None, *args, **kwargs):
        """
        Compute the total loss for the autoencoder, as a weighted sum of the individual losses.

        :param weights: LossWeights dataclass instance
        """
        super().__init__(*args, **kwargs)
        self.affinity = affinity
        self.weights = weights
        self.reconstruction_loss = ReconstructionLoss()
        self.contractive_reg = ContractiveRegularization(norm="fro")
        self.rank_penalty = DeficientRankPenalty(scale)
        self.diffeomorphism_reg1 = DiffeomorphicRegularization1(norm)
        self.tangent_bundle_reg = TangentBundleRegularization(norm="fro")
        self.drift_alignment_reg = TangentDriftAlignment2(normalize=normalize)
        self.variance_log_det_reg = VarianceLogVolume()
        self.orthogonal_coordinates = OrthogonalCoordinatesRegularization()
        self.tangent_angles_reg = TangentSpaceAnglesLoss()
        self.normal_component_reg = NormalComponentReconstructionLoss()
        self.laplacian_reg = LaplacianReg()

    def forward(self, autoencoder: AutoEncoder1, x, targets):
        """
        Compute the weighted total loss.

        :param autoencoder: The autoencoder model
        :param x: Input data to the autoencoder
        :param targets: Target data for reconstruction and regularization: (P, H, Sigma, mu)
        """
        ambient_affinity = self.affinity
        # if self.weights.laplacian_weight > 0:
        # true_projection, observed_frame, ambient_cov, ambient_drift, ambient_affinity = targets
        # elif self.weights.laplacian_weight == 0:
        true_projection, observed_frame, ambient_cov, ambient_drift = targets
        # else:
        #     raise ValueError("weights.laplacian_weight must be non-negative.")
        n, D, _ = true_projection.size()
        true_normal_proj = torch.eye(D).expand(n, D, D) - true_projection
        z = autoencoder.encoder(x)
        reconstructed_points = autoencoder.decoder(z)

        # Initialize total loss.
        total_loss = 0
        # We always compute the reconstruction loss
        reconstruction_loss = self.reconstruction_loss(reconstructed_points, x)
        total_loss += self.weights.reconstruction * reconstruction_loss

        # Check which objects we need
        need_pairwise_latent_distances = self.weights.laplacian_weight > 0
        need_decoder_jacobian = (self.weights.rank_penalty_weight > 0 or
                                 self.weights.tangent_space_error_weight > 0 or
                                 self.weights.diffeomorphism_reg1 > 0 or
                                 self.weights.decoder_contraction_weight > 0 or
                                 self.weights.variance_logdet > 0 or
                                 self.weights.orthogonal_penalty_weight > 0 or
                                 self.weights.tangent_angle_weight > 0)
        need_encoder_jacobian = (self.weights.diffeomorphism_reg1 > 0 or
                                 self.weights.encoder_contraction_weight > 0 or
                                 self.weights.tangent_drift_weight > 0)
        need_decoder_hessian = self.weights.tangent_drift_weight > 0
        need_metric_tensor = (self.weights.tangent_space_error_weight > 0 or
                              self.weights.tangent_angle_weight > 0 or
                              self.weights.variance_logdet > 0 or
                              self.weights.orthogonal_penalty_weight > 0
                              )

        decoder_jacobian = None
        encoder_jacobian = None
        decoder_hessian = None
        metric_tensor = None
        latent_distances = None
        if need_pairwise_latent_distances:
            latent_distances = pairwise_distances(z)
        if need_decoder_jacobian:
            decoder_jacobian = autoencoder.decoder_jacobian(z)
        if need_encoder_jacobian:
            encoder_jacobian = autoencoder.encoder_jacobian(x)
        if need_decoder_hessian:
            decoder_hessian = autoencoder.decoder_hessian(z)
        if need_metric_tensor and need_decoder_jacobian:
            metric_tensor = torch.bmm(decoder_jacobian.mT, decoder_jacobian)
        # Laplacian regularization:
        if self.weights.laplacian_weight > 0:
            laplacian_loss = self.laplacian_reg(ambient_affinity, latent_distances)
            total_loss += laplacian_loss
        # Deficient rank penalties
        if self.weights.rank_penalty_weight > 0:
            rank_penalty = self.rank_penalty.forward(decoder_jacobian)
            total_loss += self.weights.rank_penalty_weight * rank_penalty
        # Contractive regularization
        if self.weights.encoder_contraction_weight > 0:
            contractive_loss = self.contractive_reg(encoder_jacobian)
            total_loss += self.weights.encoder_contraction_weight * contractive_loss
        if self.weights.decoder_contraction_weight > 0:
            contractive_loss = self.contractive_reg(decoder_jacobian)
            total_loss += self.weights.decoder_contraction_weight * contractive_loss
        # Tangent Bundle regularization
        if self.weights.tangent_space_error_weight > 0:
            tangent_bundle_loss = self.tangent_bundle_reg.forward(decoder_jacobian, metric_tensor, true_projection)
            total_loss += self.weights.tangent_space_error_weight * tangent_bundle_loss
        # Drift alignment regularization
        if self.weights.tangent_drift_weight > 0:
            drift_alignment_loss = self.drift_alignment_reg.forward(encoder_jacobian,
                                                                    decoder_hessian,
                                                                    ambient_cov,
                                                                    ambient_drift,
                                                                    true_normal_proj)
            total_loss += self.weights.tangent_drift_weight * drift_alignment_loss

        # Diffeomorphism regularization
        if self.weights.diffeomorphism_reg1 > 0:
            diffeomorphism_loss1 = self.diffeomorphism_reg1(decoder_jacobian, encoder_jacobian)
            total_loss += self.weights.diffeomorphism_reg1 * diffeomorphism_loss1

        if self.weights.variance_logdet > 0:
            total_loss += self.weights.variance_logdet * self.variance_log_det_reg(metric_tensor)

        if self.weights.orthogonal_penalty_weight > 0:
            total_loss += self.weights.orthogonal_penalty_weight * self.orthogonal_coordinates(metric_tensor)

        if self.weights.tangent_angle_weight > 0:
            total_loss += self.weights.tangent_angle_weight * self.tangent_angles_reg.forward(observed_frame,
                                                                                              decoder_jacobian,
                                                                                              metric_tensor)

        if self.weights.normal_component_recon > 0:
            total_loss += self.weights.normal_component_recon * self.normal_component_reg.forward(x,
                                                                                                  reconstructed_points,
                                                                                                  observed_frame)
        return total_loss
