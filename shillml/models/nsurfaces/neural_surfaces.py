from shillml.models.ffnn import FeedForwardNeuralNet
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class FullRankLoss(nn.Module):
    def __init__(self, weight=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def forward(self, df):
        norms = torch.linalg.matrix_norm(df, ord=-2)
        return self.weight * torch.sum(torch.exp(-norms ** 2))


class HyperSurfaceLoss(nn.Module):
    def __init__(self, weight=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_rank_loss = FullRankLoss(weight)

    def forward(self, model_output: Tuple[Tensor, Tensor], targets=None):
        f, df = model_output
        manifold_constraint_loss = torch.mean(torch.linalg.vector_norm(f, ord=2, dim=1) ** 2)
        full_rank_loss = self.full_rank_loss(df)
        total_loss = manifold_constraint_loss + full_rank_loss
        return total_loss


class NeuralHyperSurface(nn.Module):
    def __init__(self, neurons, activation, intrinsic_dim=2, *args, **kwargs):
        """

        :param neurons:
        :param activation:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        activations = [activation] * len(neurons)
        self.f = FeedForwardNeuralNet(neurons, activations)
        self.extrinsic_dim = neurons[0]
        self.codim = neurons[0] - intrinsic_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.f(x)
        df = self.f.jacobian_network(x)
        return f, df

    def orthonormalized_jacobian(self, x):
        """

        :param x:
        :return:
        """
        j = self.f.jacobian_network(x)
        A = torch.bmm(j, j.mT)
        A = torch.linalg.inv(A)
        U, S, V = torch.linalg.svd(A)
        sqrt_S = torch.sqrt(S.unsqueeze(-1).expand_as(U))
        B = torch.bmm(U, sqrt_S * V.transpose(-2, -1))
        N = torch.bmm(B, j)
        return N

    def orthogonal_projection(self, x):
        """

        :param x:
        :return:
        """
        N = self.orthonormalized_jacobian(x)
        return torch.eye(self.extrinsic_dim) - torch.bmm(N.mT, N)

    def mean_curvature(self, x, N):
        """

        :param x:
        :param N:
        :return:
        """
        num_batches = x.size()[0]
        mean_curvature = torch.zeros((num_batches, self.codim, 1))
        for l in range(num_batches):
            for i in range(self.codim):
                row_div = 0.
                for j in range(self.extrinsic_dim):
                    row_div += torch.autograd.grad(N[l, i, j], x, retain_graph=True)[0][0][j]
                mean_curvature[l, i, 0] = row_div
        mean_curvature *= -0.5
        return mean_curvature

    def ito_drift(self, x):
        """

        :param x:
        :return:
        """
        N = self.orthonormalized_jacobian(x)
        mean_curve = self.mean_curvature(x, N)
        return torch.bmm(N.mT, mean_curve)

    def diffusion_coefficient(self, t, x):
        """

        :param t:
        :param x:
        :return:
        """
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(1, self.extrinsic_dim)
        p = self.orthogonal_projection(x).detach().numpy()
        return p

    def drift_coefficient(self, t, x):
        """

        :param t:
        :param x:
        :return:
        """
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(1, self.extrinsic_dim)
        mu = self.ito_drift(x).detach().numpy()
        mu = mu.reshape(self.extrinsic_dim)
        return mu


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from shillml.losses.losses import fit_model
    from shillml.sdes import SDE

    seed = None
    num_pts = 50
    neurons = [3, 32, 32, 1]
    fr_weight = 0.5
    lr = 0.001
    epochs = 20000
    weight_decay = 0.01
    tn = 3.5
    ntime = 5000
    npaths = 5
    activation = nn.Tanh()
    # Metal has some issues:
    # NotImplementedError: The operator 'aten::_linalg_svd.U' is not currently implemented for the MPS device. If you
    # want this op to be added in priority during the prototype phase of this feature, please comment
    # on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the
    # environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
    # WARNING: this will be slower than running natively on MPS.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(num_pts, 3))
    x = x / np.linalg.norm(x, axis=1, ord=2, keepdims=True)
    x = torch.tensor(x, dtype=torch.float32).to(device)

    hyp = NeuralHyperSurface(neurons, activation).to(device)
    hyp_loss = HyperSurfaceLoss(fr_weight).to(device)
    fit_model(hyp, hyp_loss, x, None, epochs=epochs, lr=lr, weight_decay=weight_decay)

    x_test = rng.normal(size=(500, 3))
    x_test = x_test / np.linalg.norm(x_test, axis=1, ord=2, keepdims=True)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    print("Test loss = " + str(hyp_loss(hyp(x_test)).item()))

    sde = SDE(hyp.drift_coefficient, hyp.diffusion_coefficient)
    sample_paths = sde.sample_ensemble(x[0, :].detach().cpu(), tn, ntime, npaths)

    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    for i in range(npaths):
        ax.plot3D(sample_paths[i, :, 0], sample_paths[i, :, 1], sample_paths[i, :, 2], c="black", alpha=0.8)
    plt.show()


