"""
    This modules verifies numerically whether SVD can recover the orthogonal projection matrix P
    from an observed covariance matrix Sigma which is implicitly assumed to be of the form
    Dphi bb^T Dphi^T.

    Assertion statements are made to check equality between P_true and P_svd and also check that they
    are idempotent.
"""

if __name__ == "__main__":
    import torch
    import sympy as sp
    from shillml.utils import process_data
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.pointclouds.dynamics import SDECoefficients
    # weights
    # Inputs
    train_seed = None
    num_points = 100
    # Boundary for point cloud
    a = -3
    b = 3
    epsilon = 1.
    bounds = [(a, b), (a, b)]
    large_bounds = [(a - epsilon, b + epsilon), (a - epsilon, b + epsilon)]
    # Flattening factors
    c1, c2 = 1, 1

    # Define the manifold
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])
    # Product
    # fuv = u*v/c1

    # Paraboloid
    fuv = (u/c1)**2+(v/c2)**2

    # # Mixture of Gaussians
    # sigma_2 = 1.
    # fuv = (0.5 * sp.exp(-((u + 0.9) ** 2 + (v + 0.9) ** 2) / (2 * sigma_2)) / (np.sqrt(2 * np.pi * sigma_2)) +
    #        0.5 * sp.exp(-((u - 0.9) ** 2 + (v - 0.9) ** 2) / (2 * sigma_2)) / (np.sqrt(2 * np.pi * sigma_2)))

    # Creating the manifold
    chart = sp.Matrix([u, v, fuv])
    manifold = RiemannianManifold(local_coordinates, chart)
    coefs = SDECoefficients()

    # BM
    local_drift = sp.Matrix([0, 0])
    local_diffusion = sp.Matrix([[1, 0], [0, 1]])

    # RBM
    # local_drift = manifold.local_bm_drift()
    # local_diffusion = manifold.local_bm_diffusion()

    # # Langevin with double well potential
    # local_drift = manifold.local_bm_drift() - 0.2 * manifold.metric_tensor().inv() * sp.Matrix(
    #     [4 * u * (u ** 2 - 1), 2 * v])
    # local_diffusion = manifold.local_bm_diffusion() * coefs.diffusion_circular()/5

    # Generate the point cloud plus dynamics observations
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
    # returns points, weights, drifts, cov, local coord
    x, _, mu, cov, local_x = cloud.generate(num_points, seed=train_seed)
    p_true = cloud.get_true_orthogonal_proj(local_x)
    p_true = torch.tensor(p_true, dtype=torch.float32)
    p_svd_estimate = process_data(x, mu, cov, d=2, return_frame=True)[3]
    # Check if the first row of the true and estimated projections are close to each other
    assert torch.allclose(p_true, p_svd_estimate, atol=1e-5), "Projection matrices are not equal"

    # Check if P^2 = P for the true orthogonal projection (idempotency check)
    p_true_squared = torch.bmm(p_true, p_true)
    assert torch.allclose(p_true, p_true_squared, atol=1e-5), "True orthogonal projection does not satisfy P^2 = P!"

    # Check if P^2 = P for the SVD estimated projection (idempotency check)
    p_svd_squared = torch.bmm(p_svd_estimate, p_svd_estimate)
    assert torch.allclose(p_svd_estimate, p_svd_squared,
                          atol=1e-5), "SVD estimated projection does not satisfy P^2 = P!"

    # Print the first row of each matrix for visual inspection
    print("True Projection, First Row:")
    print(p_true[0])
    print("SVD Estimate Projection, First Row:")
    print(p_svd_estimate[0])

    # Compute and print the Mean Squared Error (MSE) between the true and estimated projections
    print("MSE of SVD-P(x) against True-P(x):")
    mse = torch.mean(torch.linalg.matrix_norm(p_svd_estimate - p_true, ord="fro")).item()
    print(mse)
    print("All assertions passed!")

