import sympy as sp
from typing import Tuple, List
from shillml.sdes import SDE
from shillml.utils.sampler import ImportanceSampler
from shillml.diffgeo import RiemannianManifold
from matplotlib.patches import Ellipse


class PointCloud:
    def __init__(self, manifold: RiemannianManifold, bounds: List[Tuple],
                 local_drift: sp.Matrix = None, local_diffusion: sp.Matrix = None):

        """
                Initialize the PointCloud object.

                :param manifold: a RiemannianManifold object
                :param bounds: List of (min, max) tuples for each parameter
                :param local_drift: A function that takes params and returns the drift vector (optional)
                :param local_diffusion: A function that takes params and returns the diffusion matrix (optional)
        """
        self.manifold = manifold
        self.bounds = bounds
        self.dimension = len(manifold.local_coordinates)
        self.target_dim = len(manifold.chart)
        self.chart_jacobian = self.manifold.chart_jacobian()

        # Create numpy functions
        self.np_volume_measure = self.manifold.sympy_to_numpy(self.manifold.volume_density())
        self.np_phi = self.manifold.sympy_to_numpy(self.manifold.chart)
        self.np_jacobian = self.manifold.sympy_to_numpy(self.chart_jacobian)

        # Create importance sampler
        self.sampler = self._create_importance_sampler()

        # Compute drift and diffusion if provided
        if local_drift and local_diffusion:
            self.local_drift = local_drift
            self.local_diffusion = local_diffusion
            self.local_covariance = self.local_diffusion * self.local_diffusion.T
            self.np_local_drift = self.manifold.sympy_to_numpy(self.local_drift)
            self.np_local_diffusion = self.manifold.sympy_to_numpy(self.local_diffusion)
            self.np_local_covariance = self.manifold.sympy_to_numpy(self.local_covariance)

            self.extrinsic_drift = self._compute_extrinsic_drift()
            self.extrinsic_covariance = self._compute_extrinsic_covariance()
            self.np_extrinsic_drift = self.manifold.sympy_to_numpy(self.extrinsic_drift)
            self.np_extrinsic_diffusion = self.manifold.sympy_to_numpy(self.chart_jacobian * self.local_diffusion)
            self.np_extrinsic_covariance = self.manifold.sympy_to_numpy(self.extrinsic_covariance)

            self._create_sdes()

    def _create_importance_sampler(self):
        return ImportanceSampler(
            self.np_volume_measure,
            *[bound for bounds in self.bounds for bound in bounds]
        )

    def _compute_extrinsic_drift(self):
        q = sp.zeros(self.target_dim, 1)
        for i in range(self.target_dim):
            hessian = sp.hessian(self.manifold.chart[i], self.manifold.local_coordinates)
            q[i] = 0.5 * sp.trace(self.local_covariance * hessian)
        return self.chart_jacobian * self.local_drift + q

    def _compute_extrinsic_covariance(self):
        return self.chart_jacobian * self.local_covariance * self.chart_jacobian.T

    def _create_sdes(self):
        self.latent_sde = SDE(lambda t, x: self.np_local_drift(*x).reshape(self.dimension),
                              lambda t, x: self.np_local_diffusion(*x))
        self.ambient_sde = SDE(lambda t, x: self.np_extrinsic_drift(*x).reshape(self.target_dim),
                               lambda t, x: self.np_extrinsic_diffusion(*x))
        return None

    def generate(self, n: int = 1000, seed=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate points on the parameterized surface and compute extrinsic drift and covariance.

        :param n: Number of points to generate
        :param seed: random seed
        :return: Tuple of (points, weights, extrinsic_drifts, extrinsic_covariances)
        """
        param_samples, weights = self.sampler.sample(n, seed)
        points = np.array([self.np_phi(*sample) for sample in param_samples])

        if hasattr(self, 'np_extrinsic_drift') and hasattr(self, 'np_extrinsic_covariance'):
            extrinsic_drifts = np.array([self.np_extrinsic_drift(*sample) for sample in param_samples])
            extrinsic_covariances = np.array([self.np_extrinsic_covariance(*sample) for sample in param_samples])
        else:
            extrinsic_drifts = None
            extrinsic_covariances = None

        return points.squeeze(), weights, extrinsic_drifts.squeeze(), extrinsic_covariances, param_samples

    def plot_point_cloud(self, points=None, weights=None, drifts=None, plot_drift=False,
                         drift_scale=1.0, alpha=0.5, figsize=(10, 8)):
        """
        Plot the point cloud with an option to show the drift vector field.

        :param points: The points to plot (if None, generates new points)
        :param weights: The weights of the points (for sizing)
        :param drifts: The extrinsic drifts at each point
        :param plot_drift: Whether to plot the drift vector field
        :param drift_scale: Scaling factor for drift vectors
        :param alpha: Transparency of points
        :param figsize: Figure size
        """
        if points is None or weights is None or (plot_drift and drifts is None):
            points, weights, drifts, cov, param_samples = self.generate(n=1000)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Scale weights for point sizes
        sizes = 50 * weights / np.max(weights)

        # Plot points
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             s=sizes, alpha=alpha, c=weights)

        if plot_drift:
            # Plot drift vectors
            ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                      drifts[:, 0], drifts[:, 1], drifts[:, 2],
                      length=drift_scale, normalize=True, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud with Importance Weights' +
                     (' and Drift Vectors' if plot_drift else ''))

        plt.colorbar(scatter, label='Weight')
        plt.tight_layout()
        plt.show()

    def plot_point_cloud_covariance(self, points=None, covariances=None, figsize=(10, 8)):
        """
        Plot the point cloud colored by the smallest singular value of the covariance at each point.

        :param points: The points to plot (if None, generates new points)
        :param covariances: The extrinsic covariances at each point
        :param figsize: Figure size
        """
        if points is None or covariances is None:
            points, _, _, covariances, param_samples = self.generate(n=1000)

        # Compute smallest singular values
        largest_singvals = np.array([np.linalg.svd(cov, compute_uv=False)[0]
                                     for cov in covariances])

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=largest_singvals, cmap='viridis')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Colored by Largest Singular Value of Covariance')

        plt.colorbar(scatter, label='Largest Singular Value')
        plt.tight_layout()
        plt.show()

    def plot_eigenvectors(self, points=None, matrices=None, scale=1.0, figsize=(10, 8)):
        """
        Plot eigenvectors of the matrix field at each point.

        :param points: The points to plot (if None, generates new points)
        :param matrices: The matrices at each point
        :param scale: Scaling factor for eigenvectors
        :param figsize: Figure size
        """
        if points is None or matrices is None:
            points, _, _, _, param_samples = self.generate(n=1000)
            matrices = [self.np_local_diffusion(*sample) for sample in param_samples]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for point, matrix in zip(points, matrices):
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            for i in range(len(eigenvalues)):
                ax.quiver(point[0], point[1], point[2],
                          eigenvectors[0, i], eigenvectors[1, i], eigenvectors[2, i],
                          length=scale * eigenvalues[i], color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Eigenvectors of Matrix Field')
        plt.show()

    def plot_heatmap(self, points=None, matrices=None, figsize=(10, 8)):
        """
        Plot heatmap of the matrix field entries at each point.

        :param points: The points to plot (if None, generates new points)
        :param matrices: The matrices at each point
        :param figsize: Figure size
        """
        if points is None or matrices is None:
            points, _, _, _, param_samples = self.generate(n=1000)
            matrices = [self.np_local_diffusion(*sample) for sample in param_samples]

        fig, axs = plt.subplots(self.dimension, self.dimension, figsize=figsize, subplot_kw={'projection': '3d'})

        for i in range(self.dimension):
            for j in range(self.dimension):
                values = np.array([matrix[i, j] for matrix in matrices])
                scatter = axs[i, j].scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap='viridis')
                fig.colorbar(scatter, ax=axs[i, j])
                axs[i, j].set_title(f'Matrix Entry ({i}, {j})')

        plt.tight_layout()
        plt.show()

    def plot_tensor_glyphs(self, points=None, matrices=None, scale=1.0, figsize=(10, 8)):
        """
        Plot tensor glyphs to visualize the matrix field at each point.

        :param points: The points to plot (if None, generates new points)
        :param matrices: The matrices at each point
        :param scale: Scaling factor for tensor glyphs
        :param figsize: Figure size
        """
        if points is None or matrices is None:
            points, _, _, _, param_samples = self.generate(n=1000)
            matrices = [self.np_local_diffusion(*sample) for sample in param_samples]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for point, matrix in zip(points, matrices):
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            for i in range(len(eigenvalues)):
                glyph = Ellipse((point[0], point[1]), width=eigenvalues[i] * scale,
                                height=eigenvalues[(i + 1) % len(eigenvalues)] * scale,
                                angle=np.rad2deg(np.arctan2(eigenvectors[1, i], eigenvectors[0, i])))
                ax.add_patch(glyph)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Tensor Glyphs of Matrix Field')
        plt.show()

    def plot_sample_paths(self, tn=1, ntime=1000, npaths=50, figsize=(16, 8)):
        """
        Plot sample paths of the process in both 3D ambient space and 2D local coordinates.

        :param tn: Final time
        :param ntime: Number of time steps
        :param npaths: Number of paths
        :param figsize: Figure size
        """
        points, weights, extrinsic_drifts, extrinsic_covariances, param_samples = self.generate(n=1000)

        # Generate latent paths
        paths1 = self.latent_sde.sample_ensemble(param_samples[0, :], tn=tn, ntime=ntime, npaths=npaths, noise_dim=self.dimension)
        paths = np.zeros((npaths, ntime + 1, self.target_dim))
        for j in range(npaths):
            for i in range(ntime + 1):
                paths[j, i, :] = np.squeeze(self.np_phi(*paths1[j, i, :]))

        fig = plt.figure(figsize=figsize)

        # Plot 3D ambient paths
        ax1 = fig.add_subplot(121, projection='3d')
        for j in range(npaths):
            ax1.plot3D(paths[j, :, 0], paths[j, :, 1], paths[j, :, 2], alpha=0.8)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Sample Paths in 3D Ambient Space')

        # Plot 2D local coordinate paths
        ax2 = fig.add_subplot(122)
        for j in range(npaths):
            ax2.plot(paths1[j, :, 0], paths1[j, :, 1], alpha=0.8)

        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        ax2.set_title('Sample Paths in 2D Local Coordinates')

        plt.tight_layout()
        plt.show()

    def plot_drift_vector_field(self, points=None, weights=None, drifts=None, drift_scale=1.0, alpha=0.5,
                                figsize=(10, 8)):
        """
        Plot the drift vector field.

        :param points: The points to plot (if None, generates new points)
        :param weights: The weights of the points (for sizing)
        :param drifts: The extrinsic drifts at each point
        :param drift_scale: Scaling factor for drift vectors
        :param alpha: Transparency of points
        :param figsize: Figure size
        """
        if points is None or weights is None or drifts is None:
            points, weights, drifts, _, _ = self.generate(n=1000)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Scale weights for point sizes
        sizes = 50 * weights / np.max(weights)

        # Plot points
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             s=sizes, alpha=alpha, c=weights)

        # Plot drift vectors
        ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                  drifts[:, 0], drifts[:, 1], drifts[:, 2],
                  length=drift_scale, normalize=True, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drift Vector Field')

        plt.colorbar(scatter, label='Weight')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from shillml.diffgeo.diffgeo import RiemannianManifold

    tn = 1
    ntime = 1000
    npaths = 50
    # Define parameters
    u, v = sp.symbols('u v')
    r, R = 1, 3
    n = 1000

    # Define parameterizations and create RiemannianManifold objects
    sphere_manifold = RiemannianManifold(
        local_coordinates=sp.Matrix([u, v]),
        chart=sp.Matrix([r * sp.cos(v) * sp.cos(u), r * sp.cos(v) * sp.sin(u), r * sp.sin(v)])
    )

    torus_manifold = RiemannianManifold(
        local_coordinates=sp.Matrix([u, v]),
        chart=sp.Matrix([(R + r * sp.cos(v)) * sp.cos(u), (R + r * sp.cos(v)) * sp.sin(u), r * sp.sin(v)])
    )

    paraboloid_manifold = RiemannianManifold(
        local_coordinates=sp.Matrix([u, v]),
        chart=sp.Matrix([u, v, u ** 2 + v ** 2])
    )

    hyperboloid_manifold = RiemannianManifold(
        local_coordinates=sp.Matrix([u, v]),
        chart=sp.Matrix([u, v, u ** 2 - v ** 2])
    )

    # Define local drift and diffusion as sp.Matrix objects
    local_drift = sp.Matrix([-sp.sin(u) + v ** 2 - 1, sp.exp(-v)])
    local_diffusion = sp.Matrix([[5 + 0.5 * sp.cos(u), v],
                                 [0, 0.2 + 0.1 * sp.sin(v)]])

    # Create PointCloud objects
    clouds = [
        PointCloud(manifold=sphere_manifold, bounds=[(0, np.pi), (0, 2 * np.pi)],
                   local_drift=local_drift, local_diffusion=local_diffusion),
        PointCloud(manifold=torus_manifold, bounds=[(0, 2 * np.pi), (0, 2 * np.pi)],
                   local_drift=local_drift, local_diffusion=local_diffusion),
        PointCloud(manifold=paraboloid_manifold, bounds=[(-2, 2), (-2, 2)],
                   local_drift=local_drift, local_diffusion=local_diffusion),
        PointCloud(manifold=hyperboloid_manifold, bounds=[(-2, 2), (-2, 2)],
                   local_drift=local_drift, local_diffusion=local_diffusion)
    ]
    surface_names = ["Sphere", "Torus", "Paraboloid", "Hyperboloid"]
    # Create plots
    for cloud, name in zip(clouds, surface_names):
        fig, axs = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection': '3d'})
        points, weights, extrinsic_drifts, extrinsic_covariances, param_samples = cloud.generate(n=n)

        # Plot drift vector field
        sizes = 50 * weights / np.max(weights)
        scatter = axs[0].scatter(points[:, 0], points[:, 1], points[:, 2], s=sizes, alpha=0.5, c=weights)
        axs[0].quiver(points[:, 0], points[:, 1], points[:, 2],
                      extrinsic_drifts[:, 0], extrinsic_drifts[:, 1], extrinsic_drifts[:, 2],
                      length=0.5, normalize=True, color='r')
        axs[0].set_title(f'{name} - Drift Vector Field')
        plt.colorbar(scatter, ax=axs[0], label='Weight')

        # Generate latent paths:
        paths1 = cloud.latent_sde.sample_ensemble(param_samples[0, :], tn=tn, ntime=ntime, npaths=npaths, noise_dim=2)
        paths = np.zeros((npaths, ntime + 1, 3))
        for j in range(npaths):
            for i in range(ntime + 1):
                paths[j, i, :] = np.squeeze(cloud.np_phi(*paths1[j, i, :]))

        # Plot covariance coloring
        largest_singvals = np.array([np.linalg.svd(cov, compute_uv=False)[0] for cov in extrinsic_covariances])
        scatter = axs[1].scatter(points[:, 0], points[:, 1], points[:, 2], c=largest_singvals, cmap='viridis')
        for j in range(npaths):
            axs[1].plot3D(paths[j, :, 0], paths[j, :, 1], paths[j, :, 2], c="black", alpha=0.8)
        axs[1].set_title(f'{name} - Covariance Coloring')
        plt.colorbar(scatter, ax=axs[1], label='Largest Singular Value')

        for ax in axs:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        plt.tight_layout()
        plt.show()

