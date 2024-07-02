# Euclidean Neural SDE via Euler-Maruyama Conditional Gaussian MLE
# This works! for what its specified for--i.e. not proper manifolds.
import torch
import torch.nn as nn
from shillml.ffnn import FeedForwardNeuralNet


def log_det_cov(cov):
    """
    The logarithm of the determinant of a covariance matrix
    :param cov:
    :return:
    """
    det = torch.linalg.det(cov)
    return torch.log(torch.abs(det)+0.00001)


def mahalanobis_dist_sq(x, mu, cov):
    """
    The Mahalanobis distance squared

    :param x:
    :param mu:
    :param cov:
    :return:
    """
    inverse_covariance = torch.linalg.inv(cov)
    z = x - mu
    quadratic_form = torch.einsum('Ntk, Ntkl, Ntl -> Nt', z, inverse_covariance, z)
    return quadratic_form


def gaussian_nll(x, mu, cov):
    """
    The NLL of the conditional Gaussian resulting from the Euler-Maruyama scheme
    :param x:
    :param mu:
    :param cov:
    :return:
    """
    log_det_term = log_det_cov(cov)
    nll = 0.5 * (mahalanobis_dist_sq(x, mu, cov) + log_det_term)
    return nll.squeeze(-1).sum()


def euler_maruyama_nll(x, mu, cov, h):
    """
    The Euler-Maruyama NLL
    :param x: tensor of shape (N, n, d)
    :param mu: tensor of shape (N, n, d)
    :param sigma: tensor of shape (N, n, d, m)
    :param h: float
    :return:
    """
    x1 = x[:, :-1, :]
    x2 = x[:, 1:, :]
    drift = h * mu
    cov1 = h * cov
    loss = gaussian_nll(x2-x1, drift, cov1)
    return loss


def noncentral_chi_sq_log_likelihood(q, lam):
    log_f = -0.5 * (q + lam) - 0.5 * torch.log(q) + torch.log(torch.cosh(torch.sqrt(lam * q)))
    return log_f


class NeuralSDE(nn.Module):
    def __init__(self, state_dim, hidden_dim, drift_act, diffusion_act, noise_dim, *args, **kwargs):
        """
        A neural stochastic differential equation model.

        Args:
            state_dim (int): The dimensionality of the state.
            hidden_dim (list of int): The number of neurons in each hidden layer.
            drift_act (callable): The activation function for the drift neural network.
            diffusion_act (callable): The activation function for the diffusion neural network.
            noise_dim (int): The dimensionality of the noise.
        """
        super().__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        neurons_mu = [state_dim] + hidden_dim + [state_dim]
        neurons_sigma = [state_dim] + hidden_dim + [state_dim * noise_dim]
        num_activations = len(neurons_mu) - 1
        activations_mu = [drift_act for _ in range(num_activations)] + [None]
        activations_sigma = [diffusion_act for _ in range(num_activations)] + [None]
        self.drift_net = FeedForwardNeuralNet(neurons_mu, activations_mu)
        self.diffusion_net = FeedForwardNeuralNet(neurons_sigma, activations_sigma)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x1 = x[:, :-1, :]
        N = x.size(0)
        n = x.size(1)
        mu = self.drift_net(x1)
        sigma = self.diffusion_net(x1).view((N, n - 1, self.state_dim, self.noise_dim))
        cov = torch.einsum('Ntik,Ntjk->Ntij', sigma, sigma)
        return mu, cov

    def mu_fit(self, t, x):
        """

        :param t:
        :param x:
        :return:
        """
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            return self.drift_net(x).detach().numpy()

    def sigma_fit(self, t, x):
        """

        :param t:
        :param x:
        :return:
        """
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            return self.diffusion_net(x).view((self.state_dim, self.noise_dim)).detach().numpy()

    def fit(self, ensemble, lr, epochs, printfreq=100, h=1 / 252, weight_decay=0., scheme="em"):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            mu, cov = self.forward(ensemble)
            total_loss = euler_maruyama_nll(ensemble, mu, cov, h)
            total_loss.backward()

            # Monitor gradient norms
            grad_norms = {name: torch.norm(param.grad).item() for name, param in self.named_parameters() if
                          param.grad is not None}
            max_grad_norm = max(grad_norms.values(), default=0)
            min_grad_norm = min(grad_norms.values(), default=0)

            optimizer.step()

            if epoch % printfreq == 0:
                print(
                    f'Epoch: {epoch}, Train-Loss: {total_loss.item()}, Max-Grad-Norm: {max_grad_norm}, Min-Grad-Norm: {min_grad_norm}')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from shillml.ffnn import get_device
    from shillml.sdes import SDE
    from shillml.processes import get_process

    # Choose which SDE to run
    device = "gpu"
    sde_choice = "OU"  # Options: "OU", "GBM", "CIR", "circle", "sphere", "chaos"
    x00 = [1.]
    noise_dim = 1
    x0 = torch.tensor(x00, dtype=torch.float32)
    x0np = np.array(x00)
    tn = 10.
    ntime = 8000
    ntrain = 1000
    npaths = 10
    npaths_fit = 30
    seed = 17
    lr = 0.0001
    weight_decay = 0.
    epochs = 9000
    hidden_dim = [8, 8]
    printfreq = 100
    drift_act = nn.Tanh()
    diffusion_act = nn.Tanh()
    state_dim = x0.size()[0]
    h = tn / ntime
    torch.manual_seed(seed)

    # Simulate a process
    mu, sigma = get_process(sde_choice)
    sde = SDE(mu, sigma)
    ensemble = sde.sample_ensemble(x0np, tn, ntime, npaths, noise_dim=noise_dim)
    ensemble = torch.tensor(ensemble, dtype=torch.float32)
    # Train test split
    training_ensemble = torch.zeros((npaths, ntrain, state_dim))
    test_ensemble = torch.zeros((npaths, ntime - ntrain + 1, state_dim))
    for j in range(npaths):
        training_ensemble[j, :, :] = ensemble[j, :ntrain, :]
        test_ensemble[j, :, :] = ensemble[j, ntrain:, :]
    # Fit the Neural SDE
    nsde = NeuralSDE(state_dim, hidden_dim, drift_act, diffusion_act, noise_dim)
    nsde.to(device=get_device(device))
    nsde.fit(training_ensemble, lr, epochs, printfreq, h, weight_decay)
    mu_f, sigma_f = nsde(test_ensemble)
    print("NLL on Test Ensemble = " + str(euler_maruyama_nll(test_ensemble, mu_f, sigma_f, h)))

    # Generating paths
    sde_fit = SDE(nsde.mu_fit, nsde.sigma_fit)
    ensemble_fit = sde_fit.sample_ensemble(x0np, tn, ntime, npaths=npaths_fit)

    ensemble = ensemble.detach().numpy()
    mean_path = np.mean(ensemble, axis=0)
    mean_path_fit = np.mean(ensemble_fit, axis=0)
    # Compute MSE between true mean path and model mean path
    mse = np.mean(np.linalg.norm(mean_path - mean_path_fit, axis=1))
    print(f"MSE between true mean path and model mean path: {mse}")

    fig = plt.figure()
    t = np.linspace(0, tn, ntime + 1)

    if state_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(npaths):
            ax.plot(ensemble[i, :, 0], ensemble[i, :, 1], ensemble[i, :, 2], c="black", alpha=0.5)
        for i in range(npaths_fit):
            ax.plot(ensemble_fit[i, :, 0], ensemble_fit[i, :, 1], ensemble_fit[i, :, 2], c="blue", alpha=0.5)
    else:
        ax = fig.add_subplot(111)
        for i in range(npaths):
            if state_dim == 2:
                ax.plot(ensemble[i, :, 0], ensemble[i, :, 1], c="black", alpha=0.5)
            elif state_dim == 1:
                ax.plot(t, ensemble[i, :, 0], c="black", alpha=0.5)

        for i in range(npaths_fit):
            if state_dim == 2:
                ax.plot(ensemble_fit[i, :, 0], ensemble_fit[i, :, 1], c="blue", alpha=0.5)
            elif state_dim == 1:
                ax.plot(t, ensemble_fit[i, :, 0], c="blue", alpha=0.5)

    if state_dim == 1:
        ax.plot(t, mean_path, c="red", label='True Mean Path', linewidth=2)
        ax.plot(t, mean_path_fit, c="green", label='Model Mean Path', linewidth=2)
    elif state_dim == 2:
        ax.plot(mean_path[:, 0], mean_path[:, 1], c="red", label='True Mean Path', linewidth=2)
        ax.plot(mean_path_fit[:, 0], mean_path_fit[:, 1], c="green", label='Model Mean Path', linewidth=2)
    elif state_dim == 3:
        ax.plot(mean_path[:, 0], mean_path[:, 1], mean_path[:, 2], c="red", label='True Mean Path', linewidth=2)
        ax.plot(mean_path_fit[:, 0], mean_path_fit[:, 1], mean_path_fit[:, 2], c="green", label='Model Mean Path',
                linewidth=2)

    true_line = plt.Line2D([], [], color='black', label='True')
    model_line = plt.Line2D([], [], color='blue', label='Model')
    ax.legend(handles=[true_line, model_line, plt.Line2D([], [], color='red', label='True Mean Path'),
                       plt.Line2D([], [], color='green', label='Model Mean Path')])
    plt.show()
