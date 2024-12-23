import sympy as sp
import torch
from shillml.pointclouds import PointCloud
from shillml.diffgeo import RiemannianManifold


def define_manifold(surface_type="paraboloid", c1=3, c2=3):
    """
    Define a Riemannian manifold based on a given surface type.

    Args:
        surface_type (str): Type of surface ("paraboloid", "sphere", etc.).
        c1 (float): Scaling factor for the first coordinate.
        c2 (float): Scaling factor for the second coordinate.

    Returns:
        RiemannianManifold: Initialized manifold object.
    """
    u, v = sp.symbols("u v", real=True)

    if surface_type == "paraboloid":
        fuv = (u / c1) ** 2 + (v / c2) ** 2
    elif surface_type == "product":
        fuv = u * v / c1
    elif surface_type == "rational":
        fuv = (u+v)/(1+u**2+v**2)
    elif surface_type == "sphere":
        fuv = sp.Matrix([sp.sin(u) * sp.cos(v), sp.sin(u) * sp.sin(v), sp.cos(u)])
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")

    chart = sp.Matrix([u, v, fuv])
    return RiemannianManifold(sp.Matrix([u, v]), chart)


def generate_point_cloud(
    manifold, bounds, drift_type="arbitrary", num_points=30, seed=None
):
    """
    Generate a point cloud with dynamics observations on the given manifold.

    Args:
        manifold (RiemannianManifold): The manifold object.
        bounds (list): Boundary values for generating points.
        drift_type (str): Type of drift ("bm", "arbitrary", etc.).
        num_points (int): Number of points to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Generated points, drifts, covariances, and local coordinates.
    """
    # Define local drift and diffusion
    u, v = sp.symbols("u v", real=True)
    if drift_type == "bm":
        local_drift = sp.Matrix([0, 0])
        local_diffusion = sp.Matrix([[1, 0], [0, 1]])
    elif drift_type == "rbm":
        local_drift = manifold.local_bm_drift()
        local_diffusion = manifold.local_bm_diffusion()
    elif drift_type == "double well":
        double_well_potential = sp.Matrix([4 * u * (u ** 2 - 1), 2 * v])/4
        local_drift = manifold.local_bm_drift() - 0.5 * manifold.metric_tensor().inv() * double_well_potential
        local_diffusion = manifold.local_bm_diffusion()
    elif drift_type == "arbitrary":
        local_drift = sp.Matrix([u * v, -sp.sin(u)]) / 3
        local_diffusion = sp.Matrix([[u - v, u * v], [u + v, sp.sin(u) * v]]) / 2
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")

    cloud = PointCloud(
        manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True
    )

    # Generate the point cloud
    if seed is not None:
        torch.manual_seed(seed)
    points, _, drifts, covariances, local_coords = cloud.generate(num_points, seed=seed)
    return points, drifts, covariances, local_coords

def test_point_cloud_generation():
    # Define parameters
    surface_types = ["paraboloid", "product", "rational", "sphere"]
    drift_types = ["bm", "rbm", "double well", "arbitrary"]
    bounds = [(-1, 1), (-1, 1)]
    num_points = 10
    seed = 42

    # Test each surface type with each drift type
    for surface in surface_types:
        print(f"\nTesting surface type: {surface}")
        manifold = define_manifold(surface_type=surface)

        for drift in drift_types:
            print(f"  Drift type: {drift}")
            try:
                points, drifts, covariances, local_coords = generate_point_cloud(
                    manifold, bounds, drift_type=drift, num_points=num_points, seed=seed
                )
                print(f"    Points shape: {points.shape}")
                print(f"    Drifts shape: {drifts.shape}")
                print(f"    Covariances shape: {covariances.shape}")
                print(f"    Local coords shape: {local_coords.shape}")
            except ValueError as e:
                print(f"    Error: {e}")


if __name__ == "__main__":
    test_point_cloud_generation()