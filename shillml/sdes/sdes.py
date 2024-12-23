"""
    A class implementation of a SDE using numpy.
"""
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import numpy as np


def plot_sample_path(t, x, ax: plt.Axes) -> None:
    """

    :param t:
    :param x:
    :param ax:
    :return:
    """
    d = x.shape[1]
    if d == 1:
        ax.plot(t, x)
    elif d == 2:
        ax.plot(x[:, 0], x[:, 1])
    elif d == 3:
        ax.plot3D(x[:, 0], x[:, 1], x[:, 2])


def plot_ensemble(t, x, ax: plt.Axes) -> None:
    """

    :param t:
    :param x:
    :param ax:
    :return:
    """
    num_paths = x.shape[0]
    for i in range(num_paths):
        plot_sample_path(t, x[i, :, :], ax)


def find_time(t, x, a=float('-inf'), b=float('inf')):
    """
    Find the first time in time grid t such that the value of x lies in interval (a, b).

    Parameters:
    - t (list or array-like): The time grid.
    - x (list or array-like): The time-series.
    - a (float): Lower bound of the interval (default is negative infinity).
    - b (float): Upper bound of the interval (default is positive infinity).

    Returns:
    - float: The first time from t where x lies in interval (a, b) or infinity if no such time exists.
    """
    for ti, xi in zip(t, x):
        if a < xi < b:
            return ti
    return float('inf')


def ensemble_terminal_average(h, paths):
    """

    :param h: function of (x), representing the terminal cost
    :param paths: ensemble of paths solving the SDE of shape (npaths, ntime + 1, d)
    :return: float
    """
    return np.mean(h(paths[:, -1, :]))


def ensemble_running_average(f, tgrid, paths):
    """

    :param f: running cost, function of (t,x)
    :param tgrid: time grid partition of [t_0, t_n]
    :param paths: ensemble of paths of shape (npaths, ntime + 1, d)
    :return: float
    """
    npaths = paths.shape[0]
    running_cost = np.zeros(npaths)
    for i in range(npaths):
        y = f(tgrid, paths[i, :, :]).reshape(paths.shape[1])
        running_cost[i] = trapezoid(y, tgrid)
    return np.mean(running_cost)


class SDE:
    def __init__(self, mu, sigma):
        """
        Declare an SDE object. Methods include solvers for simulating sample paths and solving the
        Kolmogorov Backward Equation using conditional averages.

        :param mu: drift coefficient function of (t,x) returns shape (d, )
        :param sigma: diffusion coefficient function of (t,x) returns shape (d, n) where n is the size
        of the driving Brownian motion noise.

        Attributes:
            mu:     the infinitesimal drift coefficient function of (t,x)
            sigma:  the infinitesimal diffusion coefficient function of (t,x)

        Methods:
            solve: solve the SDE numerically using the Euler-Maruyama scheme
            sample_ensemble: generate an ensemble of sample paths starting from the same
            point by repeatedly calling .solve(...)
            mc_pde_solve: Solve the Kolmogorov Backward PDE at a single point (t,x) using Monte-Carlo estimation
        via the Feynman-Kac formula.
            solve_and_stop: solve the SDE stopping at a boundary condition


        """
        self.mu = mu
        self.sigma = sigma

    def solve(self, x0, tn, ntime=100, t0=0., seed=None, noise_dim=None):
        """
        Solve a generic SDE using Euler-Maruyama

        :param x0: initial state
        :param tn: terminal time horizon
        :param ntime: number of time-steps
        :param t0: initial time
        :param seed: pseudo RNG seed
        :param noise_dim: optional noise dimension (if none, use state dim)
        :return: array of shape (ntime+1, d) where d is state dim
        """
        rng = np.random.default_rng(seed)
        if len(x0.shape) == 0:
            x0 = np.array([x0])
        d = x0.shape[0]
        if noise_dim is None:
            noise_dim = d
        x = np.zeros((ntime+1, d))
        x[0, :] = x0
        h = (tn-t0)/ntime
        for i in range(ntime):
            drift = self.mu(t0 + i * h, x[i, :])
            diffusion = self.sigma(t0 + i * h, x[i, :])
            db = rng.normal(scale=np.sqrt(h), size=noise_dim).reshape(noise_dim, 1)
            x[i + 1, :] = x[i, :] + drift * h + (diffusion @ db).reshape(d)
        return x

    def sample_ensemble(self, x0, tn, ntime=100, npaths=5, t0=0., noise_dim=None):
        """

        :param x0: initial state
        :param tn: terminal time horizon
        :param ntime: number of time-steps
        :param npaths: number of paths
        :param t0: initial time
        :param noise_dim: optional noise dimension (if none, use state dim)
        :return: array of shape (npaths, ntime+1, d) where d is state dim
        """
        if len(x0.shape) == 0:
            x0 = np.array([x0])
        d = x0.shape[0]
        x = np.zeros((npaths, ntime + 1, d))
        for i in range(npaths):
            x[i, :, :] = self.solve(x0, tn, ntime, t0, None, noise_dim)
        return x

    def mc_pde_solve(self, t, x, f, h, tn, ntime=100, npaths=5, noise_dim=None):
        """
        Solve the Kolmogorov Backward PDE at a single point (t,x) using Monte-Carlo estimation
        via the Feynman-Kac formula.

        :param t: time input for the solution u(t, x)
        :param x: state input for the solution u(t, x)
        :param f: the running cost function of (t, x)
        :param h: the terminal cost function of (x)
        :param tn: the terminal time value
        :param ntime: number of time-steps
        :param npaths: number of paths used in the MC estimation
        :param noise_dim: noise dimension of the driving Brownian motion
        :return:
        """
        tgrid = np.linspace(t, tn, ntime+1)
        paths = self.sample_ensemble(x, tn, ntime, npaths, t, noise_dim)
        terminal_cost = ensemble_terminal_average(h, paths)
        running_cost = ensemble_running_average(f, tgrid, paths)
        u = running_cost + terminal_cost
        return u

    def solve_and_stop(self, a, b, x0, tn, ntime=100, t0=0., seed=None, noise_dim=None):
        """
        Solve a generic SDE using Euler-Maruyama

        :param a: lower bound of box [a,b]^d to stop at
        :param b: upper bound of box [a,d]^d to stop at
        :param x0: initial state
        :param tn: terminal time horizon
        :param ntime: number of time-steps
        :param t0: initial time
        :param seed: pseudo RNG seed
        :param noise_dim: optional noise dimension (if none, use state dim)
        :return: array of shape (ntime+1, d) where d is state dim
        """
        rng = np.random.default_rng(seed)
        if len(x0.shape) == 0:
            x0 = np.array([x0])
        d = x0.shape[0]
        if noise_dim is None:
            noise_dim = d
        x = np.zeros((ntime + 1, d))
        x[0, :] = x0
        h = (tn - t0) / ntime
        for i in range(ntime):
            drift = self.mu(t0 + i * h, x[i, :])
            diffusion = self.sigma(t0 + i * h, x[i, :])
            db = rng.normal(scale=np.sqrt(h), size=noise_dim).reshape(noise_dim, 1)
            x[i + 1, :] = x[i, :] + drift * h + (diffusion @ db).reshape(d)
            if np.all(x[i + 1, :] > a) and np.all(x[i + 1, :] < b):
                x[(i+1):, :] = x[i+1, :]
                return x, t0+h*(i+1)
        return x, np.inf

    def feynman_kac_1d(self, f, h, x0, tn, grid_bds, grid_sizes, ntime, npaths, noise_dim=None):
        """

        :param f:
        :param h:
        :param x0:
        :param tn:
        :param grid_bds:
        :param grid_sizes:
        :param ntime:
        :param npaths:
        :param noise_dim:
        :return:
        """
        a = grid_bds[0]
        b = grid_bds[1]
        space_grid_size = grid_sizes[0]
        time_grid_size = grid_sizes[1]
        # Compute solution over a grid (t,x)
        x_grid = np.linspace(a, b, space_grid_size)
        t_grid = np.linspace(0., tn, time_grid_size)
        u = np.zeros((time_grid_size, space_grid_size))
        for i in range(time_grid_size):
            for j in range(space_grid_size):
                u[i, j] = self.mc_pde_solve(t_grid[i], x_grid[j], f, h, tn, ntime, npaths, noise_dim)

        # Create mesh for plotting
        t_mesh, x_mesh = np.meshgrid(t_grid, x_grid, indexing="ij")
        # Generating an ensemble of sample paths for plotting
        path_time_grid = np.linspace(0., tn, ntime + 1)
        paths = self.sample_ensemble(x0, tn, ntime, npaths, 0., noise_dim)

        # Plot the solution u(t,x) and an ensemble of paths
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        # Plot the ensemble
        plot_ensemble(path_time_grid, paths, ax1)
        ax1.set_xlabel("t")
        ax1.set_ylabel("x(t)")
        # Plot the PDE solution
        ax2.remove()
        ax2 = plt.subplot(1, 2, 2, projection="3d")
        sc = ax2.plot_surface(t_mesh, x_mesh, u, cmap="viridis")
        fig.colorbar(sc, ax=ax2, label="u(t,x)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("x")
        # Finally, show everything!
        plt.show()

    def feynman_kac_2d(self, f, h, x0, tn, grid_bds, grid_sizes, ntime, npaths, noise_dim=None):
        """

        :param f:
        :param h:
        :param x0:
        :param tn:
        :param grid_bds:
        :param grid_sizes:
        :param ntime:
        :param npaths:
        :param noise_dim:
        :return:
        """
        a = grid_bds[0]
        b = grid_bds[1]
        c = grid_bds[2]
        d = grid_bds[3]
        space_grid_size = grid_sizes[0]
        time_grid_size = grid_sizes[1]
        # Compute solution over a grid (t,x)
        x_grid = np.linspace(a, b, space_grid_size)
        y_grid = np.linspace(c, d, space_grid_size)
        t_grid = np.linspace(0., tn, time_grid_size)
        u = np.zeros((time_grid_size, space_grid_size, space_grid_size))
        for i in range(time_grid_size):
            for j in range(space_grid_size):
                for k in range(space_grid_size):
                    x1 = np.array([x_grid[j], y_grid[k]])
                    u[i, j, k] = self.mc_pde_solve(t_grid[i], x1, f, h, tn, ntime, npaths, noise_dim)

        # Create mesh for plotting
        t_mesh, x_mesh, y_mesh = np.meshgrid(t_grid, x_grid, y_grid, indexing="ij")
        # Generating an ensemble of sample paths for plotting
        path_time_grid = np.linspace(0., tn, ntime + 1)
        paths = self.sample_ensemble(x0, tn, ntime, npaths, 0., noise_dim)

        # Plot the solution u(t,x) and an ensemble of paths
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 7))
        # Plot the ensemble
        plot_ensemble(path_time_grid, paths, ax1)
        ax1.set_xlabel("x(t)")
        ax1.set_ylabel("y(t)")
        # Plot the PDE solution at time 0
        ax2.remove()
        ax2 = plt.subplot(2, 2, 2, projection="3d")
        sc = ax2.plot_surface(x_mesh[0, :, :], y_mesh[0, :, :], u[0, :, :], cmap="viridis")
        fig.colorbar(sc, ax=ax2, label="u(t,x)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("t=" + str(0.))

        # Plot the PDE solution at time +1
        ax3.remove()
        ax3 = plt.subplot(2, 2, 3, projection="3d")
        sc = ax3.plot_surface(x_mesh[2, :, :], y_mesh[2, :, :], u[2, :, :], cmap="viridis")
        fig.colorbar(sc, ax=ax3, label="u(t,x)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_title("t=" + str(2 * (tn) / time_grid_size))

        # Plot the PDE solution at time +1
        ax4.remove()
        ax4 = plt.subplot(2, 2, 4, projection="3d")
        sc = ax4.plot_surface(x_mesh[-1, :, :], y_mesh[-1, :, :], u[-1, :, :], cmap="viridis")
        fig.colorbar(sc, ax=ax4, label="u(t,x)")
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        ax4.set_title("t=" + str(tn))
        # Finally, show everything!
        plt.show()


if __name__ == "__main__":
    # A template for 1d Fenyman-Kac problems (solving PDEs with MC estimates of SDEs)
    tn = 0.1
    ntime = 50
    npaths = 100
    noise_dim = None
    x0 = np.array([1.])
    a = -1.5
    b = 1.5
    space_grid_size = 25
    time_grid_size = 10
    grid_bds = [a, b]
    grid_sizes = [space_grid_size, time_grid_size]


    def mu(t, x):
        return -x


    def sigma(t, x):
        return np.eye(1)*0.1


    def f(t, x):
        # return np.abs(x) < 1.
        return np.zeros(t.shape)


    def h(x):
        return np.abs(x) > 1.


    # For 1-d PDE estimation
    sde = SDE(mu, sigma)
    sde.feynman_kac_1d(f, h, x0, tn, grid_bds, grid_sizes, ntime, npaths, noise_dim)

    # A template for 2d Fenyman-Kac problems (solving PDEs with MC estimates of SDEs)
    tn = 0.1
    ntime = 5
    npaths = 100
    noise_dim = None
    x0 = np.array([0., 0.])
    a = -2.5
    b = 2.5
    c = -2.5
    d = 2.5
    space_grid_size = 20
    time_grid_size = 5
    grid_bds = [a, b, c, d]
    grid_sizes = [space_grid_size, time_grid_size]


    def mu(t, x):
        return np.array([0., 0.])


    def sigma(t, x):
        return np.eye(2)


    def f(t, x):
        # return np.abs(x) < 1.
        return (np.linalg.norm(x, axis=1) < 1.) / tn
        # return np.zeros(t.shape)


    def h(x):
        # return np.linalg.norm(x, ord=2) > 1.
        return 0.


    # For 2-d PDE estimation
    sde = SDE(mu, sigma)
    sde.feynman_kac_2d(f, h, x0, tn, grid_bds, grid_sizes, ntime, npaths, noise_dim)