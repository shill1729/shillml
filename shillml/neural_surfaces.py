from shillml.ffnn import FeedForwardNeuralNet
import torch.nn as nn
import torch
import time


class NeuralHyperSurface(nn.Module):
    """

    """

    def __init__(self, neurons, activations, intrinsic_dim=2, *args, **kwargs):
        """

        :param neurons:
        :param activations:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.f = FeedForwardNeuralNet(neurons, activations)
        self.extrinsic_dim = neurons[0]
        self.codim = neurons[0] - intrinsic_dim

    def manifold_constraint_loss(self, x):
        """

        :param x:
        :return:
        """
        if len(x.size()) == 2:
            norm_square = torch.linalg.vector_norm(self.f(x), ord=2, dim=1) ** 2
            return torch.mean(norm_square)
        else:
            raise ValueError("Input 'x' should be a 2d array/tensor.")

    def full_rank_regularization(self, x, reg=0.):
        """

        :param x:
        :param reg:
        :return:
        """
        if reg == 0.:
            return 0.
        df = self.f.jacobian_network(x)
        norms = torch.linalg.matrix_norm(df, ord=-2)
        return reg * torch.sum(torch.exp(-norms ** 2))

    def loss(self, x, regs=None):
        """


        :param x:
        :param regs:
        :return:
        """
        if regs is None:
            regs = [0]
        loss1 = self.manifold_constraint_loss(x)
        loss2 = self.full_rank_regularization(x, regs[0])
        return loss1 + loss2

    def fit(self, x, lr, epochs, regs=None, print_freq=100) -> None:
        """
        Train the autoencoder on a point-cloud.

        :param x: the training data, expected to be of (N, n+1, D)
        :param lr, the learning rate
        :param epochs: the number of training epochs
        :param regs: regularization constant
        :param print_freq: print frequency of the training loss

        :return: None
        """
        start = time.time()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        for epoch in range(epochs + 1):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            total_loss = self.loss(x, regs)
            # Stepping through optimizer
            total_loss.backward()
            optimizer.step()
            if epoch % print_freq == 0:
                print('Epoch: {}: Train-Loss: {}'.format(epoch, total_loss.item()))
        end = time.time()
        print("Training time = " + str(end - start))

    def orthonormalized_jacobian(self, x):
        """

        :param x:
        :return:
        """
        j = self.f.jacobian_network(x)
        A = torch.bmm(j, j.mT)
        A = torch.linalg.inv(A)
        # B = torch.linalg.cholesky(A)
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
