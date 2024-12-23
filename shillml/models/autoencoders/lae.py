# Laplacian auto encoder
import torch
from torch import Tensor
from time import time


def pairwise_distances(X: Tensor):
    """ Returns square distances"""
    sum_x = sum(torch.pow(X, 2), 1)
    dist_matrix_sq = (-2 * torch.mm(X.T, X) + sum_x).mT + sum_x
    return dist_matrix_sq


# Define the kernel function
def kernel(dist: Tensor, sigma: float):
    return torch.exp(-dist * dist / (2 * sigma ** 2))


# Compute the weight matrix W
def computeW(dist: Tensor, sigma: float):
    N = dist.shape[0]
    W = torch.zeros((N, N), dtype=dist.dtype, device=dist.device)
    for i in range(N):
        W[i, :] = kernel(dist[i, :], sigma)
    W.fill_diagonal_(0.0)  # Ensure zero diagonal for W
    return W


# Compute the diagonal degree matrix D
def computeD(W: Tensor):
    D = torch.diag(torch.sum(W, dim=1))
    return D


# Compute D^p for diagonal matrix D
def dPow(D: Tensor, p: float = 1):
    """Given the diagonal matrix D and a power p,
    this returns D^p."""
    D_pow = torch.diag(D.diag() ** p)
    return D_pow


# Construct L-symmetric from the weight matrix W
def makeLsym(W: Tensor, D: Tensor):
    """Constructs L-symmetric from the weight matrix W."""
    L = D - W
    Dhalf = dPow(D, p=-0.5)
    return Dhalf @ L @ Dhalf


# Compute the Laplacian eigenmap
def laplacian_eigenmap(dist: Tensor, sigma: float):
    W = computeW(dist, sigma)
    D = computeD(W)
    Lsym = makeLsym(W, D)
    eigwLsym, eigvLsym = torch.linalg.eigh(Lsym)
    ordersLsym = torch.argsort(eigwLsym)
    eigwLsym = eigwLsym[ordersLsym]
    eigvLsym = eigvLsym[:, ordersLsym]
    d_half = torch.diag(1.0 / torch.sqrt(D.diag()))
    eigvLsym = d_half @ eigvLsym
    return eigwLsym, eigvLsym


# Find largest spectral gap of sorted increasing eigenvalues
def spectral_gap(eigw: Tensor):
    """ Assume the eigenvalues are increasing."""
    return torch.argmax(torch.diff(eigw)).item() + 1


# Compute the low-dimensional embedding for Laplacian Eigenmaps
def laplacian_eigenmap_embedding(dist: Tensor, sigma: float, d: int):
    eigw, eigv = laplacian_eigenmap(dist, sigma)
    return eigv[:, 1:(d + 1)]
