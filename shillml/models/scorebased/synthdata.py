import numpy as np
import matplotlib.pyplot as plt
from shillml.sdes import SDE
class SyntheticDataGenerator:
    def __init__(self, sde):
        """
        Initialize the Synthetic Data Generator with a given SDE.

        :param sde: An instance of the SDE class.
        """
        self.sde = sde

    def generate_data(self, x0, tn, ntime=100, npaths=5):
        """
        Generate synthetic data by sampling from the SDE.

        :param x0: Initial state.
        :param tn: Terminal time horizon.
        :param ntime: Number of time-steps.
        :param npaths: Number of sample paths.
        :return: Array of shape (npaths, ntime+1, d) where d is the state dimension.
        """
        return self.sde.sample_ensemble(x0, tn, ntime, npaths)

    @staticmethod
    def plot_paths(paths, title='Sample Paths'):
        """
        Plot the sample paths.

        :param paths: Array of shape (npaths, ntime+1, d) containing sample paths.
        :param title: Title of the plot.
        """
        npaths, ntime, d = paths.shape
        t = np.linspace(0, 1, ntime)
        for i in range(npaths):
            for j in range(d):
                plt.plot(t, paths[i, :, j], label=f'Path {i+1} - Dimension {j+1}')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# Define well-known stochastic processes
def brownian_motion(t, x):
    return np.zeros_like(x)

def brownian_diffusion(t, x):
    return np.eye(x.shape[0])

def geometric_brownian_motion_drift(t, x, mu=0.1):
    return mu * x

def geometric_brownian_motion_diffusion(t, x, sigma=0.2):
    return sigma * np.diag(x)

def ornstein_uhlenbeck_drift(t, x, theta=0.7, mu=0.0):
    return theta * (mu - x)

def ornstein_uhlenbeck_diffusion(t, x, sigma=0.3):
    return sigma * np.eye(x.shape[0])

# Example usage
if __name__ == "__main__":
    # Brownian Motion in 1D, 2D, and 3zD
    bm_sde = SDE(brownian_motion, brownian_diffusion)
    bm_generator = SyntheticDataGenerator(bm_sde)

    bm_paths_1d = bm_generator.generate_data(np.array([0.0]), tn=1.0, ntime=100, npaths=5)
    bm_generator.plot_paths(bm_paths_1d, title='Brownian Motion in 1D')

    bm_paths_2d = bm_generator.generate_data(np.array([0.0, 0.0]), tn=1.0, ntime=100, npaths=5)
    bm_generator.plot_paths(bm_paths_2d, title='Brownian Motion in 2D')

    bm_paths_3d = bm_generator.generate_data(np.array([0.0, 0.0, 0.0]), tn=1.0, ntime=100, npaths=5)
    bm_generator.plot_paths(bm_paths_3d, title='Brownian Motion in 3D')

    # Geometric Brownian Motion in 1D, 2D, and 3D
    gbm_sde = SDE(geometric_brownian_motion_drift, geometric_brownian_motion_diffusion)
    gbm_generator = SyntheticDataGenerator(gbm_sde)

    gbm_paths_1d = gbm_generator.generate_data(np.array([1.0]), tn=1.0, ntime=100, npaths=5)
    gbm_generator.plot_paths(gbm_paths_1d, title='Geometric Brownian Motion in 1D')

    gbm_paths_2d = gbm_generator.generate_data(np.array([1.0, 1.0]), tn=1.0, ntime=100, npaths=5)
    gbm_generator.plot_paths(gbm_paths_2d, title='Geometric Brownian Motion in 2D')

    gbm_paths_3d = gbm_generator.generate_data(np.array([1.0, 1.0, 1.0]), tn=1.0, ntime=100, npaths=5)
    gbm_generator.plot_paths(gbm_paths_3d, title='Geometric Brownian Motion in 3D')

    # Ornstein-Uhlenbeck Process in 1D, 2D, and 3D
    ou_sde = SDE(ornstein_uhlenbeck_drift, ornstein_uhlenbeck_diffusion)
    ou_generator = SyntheticDataGenerator(ou_sde)

    ou_paths_1d = ou_generator.generate_data(np.array([0.0]), tn=1.0, ntime=100, npaths=5)
    ou_generator.plot_paths(ou_paths_1d, title='Ornstein-Uhlenbeck Process in 1D')

    ou_paths_2d = ou_generator.generate_data(np.array([0.0, 0.0]), tn=1.0, ntime=100, npaths=5)
    ou_generator.plot_paths(ou_paths_2d, title='Ornstein-Uhlenbeck Process in 2D')

    ou_paths_3d = ou_generator.generate_data(np.array([0.0, 0.0, 0.0]), tn=1.0, ntime=100, npaths=5)
    ou_generator.plot_paths(ou_paths_3d, title='Ornstein-Uhlenbeck Process in 3D')
