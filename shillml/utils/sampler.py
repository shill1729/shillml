from scipy.integrate import quad, dblquad
from typing import Callable, Tuple
import numpy as np


class ImportanceSampler:
    def __init__(self, h: Callable, *bounds):
        """
        Initialize importance sampler for 1D or 2D sampling.

        Args:
            h: Target distribution (unnormalized)
            *bounds: Sequence of bounds (x_min, x_max) for 1D or (x_min, x_max, y_min, y_max) for 2D
        """
        self.h = h
        self.bounds = list(zip(bounds[::2], bounds[1::2]))
        self.dimension = len(self.bounds)

        if self.dimension not in [1, 2]:
            raise ValueError("Only 1D and 2D sampling are supported")

        self.Z = self._compute_normalization_constant()

    def _compute_normalization_constant(self):
        if self.dimension == 1:
            return quad(self.h, self.bounds[0][0], self.bounds[0][1])[0]
        else:  # self.dimension == 2
            return dblquad(self.h,
                           self.bounds[0][0], self.bounds[0][1],
                           lambda x: self.bounds[1][0],
                           lambda x: self.bounds[1][1])[0]

    def f(self, *args):
        """Normalized target distribution"""
        return self.h(*args) / self.Z

    def q(self, *args):
        """Proposal distribution (uniform)"""
        return 1 / np.prod([b[1] - b[0] for b in self.bounds])

    def sample(self, n_samples: int, seed=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate weighted samples from the target distribution.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Tuple of (samples, weights) where samples has shape (n_samples, dimension)
            and weights has shape (n_samples,)
        """
        rng = np.random.default_rng(seed)
        samples = rng.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(n_samples, self.dimension)
        )
        weights = self.f(*samples.T) / self.q(*samples.T)
        weights /= np.sum(weights)
        return samples, weights


# Example usage for 1D:
if __name__ == "__main__":
    # 1D example
    def h1d(x):
        return np.exp(-(x - 2) ** 2)


    sampler_1d = ImportanceSampler(h1d, 0, 4)  # Sample in range [0, 4]
    samples_1d, weights_1d = sampler_1d.sample(1000)
    print(samples_1d.shape)

    # 2D example
    def h2d(x, y):
        return np.exp(-(x - 2) ** 2 - (y - 2) ** 2)


    sampler_2d = ImportanceSampler(h2d, 0, 4, 0, 4)  # Sample in range [0, 4] x [0, 4]
    samples_2d, weights_2d = sampler_2d.sample(1000)
    print(samples_2d.shape)