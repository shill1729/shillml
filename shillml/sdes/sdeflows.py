import torch


def flow(points, t, W, mu_fn, sigma_fn, num_steps=1000):
    """
    Compute the stochastic flow in arbitrary dimensions using PyTorch.

    Args:
        points: torch.Tensor of shape (n_points, d) where d is the dimension of each point
        t: float, the total time
        W: torch.Tensor of shape (num_steps + 1, k) representing k-dimensional Brownian motion
        mu_fn: Callable that takes a (n_points, d) tensor and returns a (n_points, d) tensor
        sigma_fn: Callable that takes a (n_points, d) tensor and returns a (n_points, d, k) tensor
        num_steps: int, number of time steps

    Returns:
        paths: torch.Tensor of shape (num_steps + 1, n_points, d) containing the paths
    """
    dt = t / num_steps
    n_points, d = points.shape
    k = W.shape[1]  # dimension of the Brownian motion

    # Initialize paths tensor
    paths = torch.zeros(num_steps + 1, n_points, d, device=points.device, dtype=points.dtype)
    paths[0] = points

    # Compute the flow
    for i in range(num_steps):
        # Compute Brownian motion increment
        dW = W[i + 1] - W[i]  # Shape: (k,)

        # Current points
        current_points = paths[i]  # Shape: (n_points, d)

        # Compute drift and diffusion terms
        mu = mu_fn(current_points)  # Shape: (n_points, d)
        sigma = sigma_fn(current_points)  # Shape: (n_points, d, k)

        # Update points using the SDE
        # mu * dt shape: (n_points, d)
        # (sigma @ dW) shape: (n_points, d, k) @ (k,) -> (n_points, d)
        paths[i + 1] = current_points + mu * dt + torch.einsum('ijk,k->ij', sigma, dW)

    return paths
