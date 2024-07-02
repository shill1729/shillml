# Define various SDEs
import numpy as np


def get_process(sde_choice):
    if sde_choice == "OU":
        mu = ou_process_mu
        sigma = ou_process_sigma
    elif sde_choice == "GBM":
        mu = gbm_mu
        sigma = gbm_sigma
    elif sde_choice == "CIR":
        mu = cir_mu
        sigma = cir_sigma
    elif sde_choice == "circle":
        mu = circle_mu
        sigma = circle_sigma
    elif sde_choice == "sphere":
        mu = sphere_mu
        sigma = sphere_sigma
    elif sde_choice == "chaos":
        mu = chaos_mu
        sigma = chaos_sigma
    else:
        raise ValueError("Model not implemented.")
    return mu, sigma


def ou_process_mu(t, x, theta=0.5, mu=0.0):
    return theta * (mu - x)


def ou_process_sigma(t, x, sigma=0.1):
    return sigma * np.eye(x.shape[0])


def gbm_mu(t, x, alpha=0.1):
    return alpha * x


def gbm_sigma(t, x, beta=0.2):
    return beta * x


def cir_mu(t, x, kappa=0.5, theta=0.04):
    return kappa * (theta - x)


def cir_sigma(t, x, sigma=0.1):
    return sigma * np.sqrt(x)


def circle_mu(t, x):
    return - 0.5 * x / np.linalg.norm(x, ord=2) ** 2


def circle_sigma(t, x):
    return np.eye(2) - np.outer(x, x) / np.linalg.norm(x, ord=2) ** 2


def sphere_mu(t, x):
    return - x / np.linalg.norm(x, ord=2) ** 2


def sphere_sigma(t, x):
    return np.eye(3) - np.outer(x, x) / np.linalg.norm(x, ord=2) ** 2


# Drift function
def chaos_mu(t, x):
    # Arbitrarily absurd function
    return np.sin(x) + np.log(np.abs(x) + 1) - np.exp(-x ** 2) + np.tan(x / (np.abs(x) + 0.1))


# Diffusion function
def chaos_sigma(t, x):
    # Equally ridiculous
    return np.sqrt(np.abs(np.cos(x) + x ** 2)) + 0.1 * np.sign(np.sin(5 * x))
