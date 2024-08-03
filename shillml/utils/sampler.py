from scipy.integrate import dblquad
from typing import Callable, Tuple
import numpy as np


class ImportanceSampler:
    def __init__(self, h: Callable, *bounds):
        self.h = h
        self.bounds = list(zip(bounds[::2], bounds[1::2]))
        self.dimension = len(self.bounds)
        self.Z = self._compute_normalization_constant()

    def _compute_normalization_constant(self):
        if self.dimension == 2:
            return dblquad(self.h, self.bounds[0][0], self.bounds[0][1],
                           lambda x: self.bounds[1][0], lambda x: self.bounds[1][1])[0]
        else:
            raise NotImplementedError("Only 2D sampling is currently supported")

    def f(self, *args):
        return self.h(*args) / self.Z

    def q(self, *args):
        return 1 / np.prod([b[1] - b[0] for b in self.bounds])

    def sample(self, n_samples: int, seed=None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        samples = rng.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(n_samples, self.dimension)
        )
        weights = self.f(*samples.T) / self.q(*samples.T)
        weights /= np.sum(weights)
        return samples, weights
