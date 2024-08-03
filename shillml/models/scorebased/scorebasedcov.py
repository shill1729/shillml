from shillml.models.scorebased.scorebased import ScoreModel
from shillml.models import FeedForwardNeuralNet
from torch import Tensor
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from shillml.losses import MatrixMSELoss

class ScoreBasedDiffusion(ScoreModel):
    def __init__(self, input_dim, hidden_dims, score_act=F.tanh, cov_act=F.tanh, hutchinson_samples=0, *args, **kwargs):
        # Neural network for covariance
        super().__init__(input_dim, hidden_dims, score_act, hutchinson_samples, *args, **kwargs)
        cov_dims = [input_dim] + hidden_dims + [input_dim ** 2]
        cov_activations = [cov_act] * (len(cov_dims) - 2) + [None]
        self.covariance_net = FeedForwardNeuralNet(cov_dims, cov_activations)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        score = self.score_net.forward(x)
        # Compute the squared L2 norm of the score
        # score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)
        if self.hutchinson_samples > 0:
            jacobian_trace = self.hutchinson_trace(x, num_samples=self.hutchinson_samples)
        else:
            jacobian = self.score_net.jacobian_network(x)
            jacobian_trace = torch.sum(torch.diagonal(jacobian, dim1=1, dim2=2), dim=1)
        d = self.score_net.output_dim
        cov = self.covariance_net(x).view((x.size(0), d, d))
        cov_div = self.covariance_divergence(x, cov)
        return score, jacobian_trace, cov, cov_div

    @staticmethod
    def covariance_divergence(x: Tensor, cov: Tensor) -> Tensor:
        batch_size, d, _ = cov.size()
        cov_div = torch.zeros(batch_size, d, device=x.device)
        for i in range(d):
            for j in range(d):
                grad_cov = torch.autograd.grad(outputs=cov[:, i, j], inputs=x,
                                               grad_outputs=torch.ones_like(cov[:, i, j]),
                                               retain_graph=True, create_graph=True)[0]
                cov_div[:, i] += grad_cov[:, j]
        return cov_div


class ScoreBasedMatchingDiffusionLoss(nn.Module):
    def __init__(self, covariance_weight=1., stationary_weight=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.covariance_weight = covariance_weight
        self.stationary_weight = stationary_weight
        self.matrix_mse = MatrixMSELoss()

    def forward(self, model_output: Tensor, targets: Tensor):
        score, jacobian_trace, model_cov, model_cov_div = model_output
        mu, cov = targets
        # Compute the squared L2 norm of the score
        score_norm = torch.mean(torch.linalg.vector_norm(score, ord=2, dim=1) ** 2)
        jacobian_trace = torch.mean(jacobian_trace)
        score_loss = score_norm + 2 * jacobian_trace
        # Covariance loss
        covariance_loss = self.covariance_weight * self.matrix_mse.forward(model_cov, cov)
        # stationary loss
        cov_inv = torch.linalg.inv(cov)
        vec = 2*mu-model_cov_div
        stationary_target = torch.bmm(cov_inv, vec.unsqueeze(2)).squeeze()
        stationary_loss = torch.linalg.vector_norm(score-stationary_target, ord=2, dim=1)
        stationary_loss = self.stationary_weight * torch.mean(stationary_loss)
        total_loss = score_loss + covariance_loss + stationary_loss
        return total_loss
