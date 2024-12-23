"""
The purpose of this module, vanilla_ae_reg_stats.py, is to compute statistics of losses/penalties on testing sets after training
a vanilla autoencoder.

This will give a measure of the expressiveness/performativity of the vanilla autoencoder, as since we only minimize
the reconstruction loss, we can see which other penalities are automatically minimized additionally without explicitly
doing so in the gradient descent.
"""

import logging

import numpy as np
import pandas as pd
import scipy.stats as spstats
import sympy as sp
import torch.nn as nn

from shillml.diffgeo import RiemannianManifold
from shillml.losses.loss_modules import TotalLoss, LossWeights
from shillml.models.autoencoders import AutoEncoder1
from shillml.pointclouds import PointCloud
from shillml.utils import fit_model2, process_data, compute_test_losses

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global constants (Magic numbers)
ALPHA = 0.05
NUM_TRAIN_POINTS = 50
BATCH_SIZE = NUM_TRAIN_POINTS // 2
NUM_TEST_POINTS = 200
HIDDEN_LAYERS = [32]
NUM_TRIALS = 50  # Number of trials for each surface
LEARNING_RATE = 0.01
NUM_EPOCHS = 20000
A, B = -2, 2  # Bounds for surface
C1, C2 = 5., 5.  # Scaling factors
EPSILON = 1.
INPUT_DIM, LATENT_DIM = 3, 2
ENCODER_ACT = nn.Tanh()
DECODER_ACT = nn.Tanh()


# Function for surface definitions
def get_surfaces():
    return [
        lambda u, v, c1, c2: 0.1 * (u + v) / (1 + u ** 2 + v ** 2),  # Rational function
        lambda u, v, c1, c2: u * v / c1,  # Product
        lambda u, v, c1, c2: (u / c1) ** 2 + (v / c2) ** 2,  # Paraboloid
        lambda u, v, c1, c2: (u / c1) ** 2 - (v / c2) ** 2,  # Hyperbolic paraboloid
        lambda u, v, c1, c2: (  # Mixture of Gaussians
            0.5 * sp.exp(-((u + 0.9) ** 2 + (v + 0.9) ** 2) / (2 * 0.8)) / (np.sqrt(2 * np.pi * 0.8)) +
            0.5 * sp.exp(-((u - 0.9) ** 2 + (v - 0.9) ** 2) / (2 * 0.8)) / (np.sqrt(2 * np.pi * 0.8))
        )
    ]


# Function for hypothesis testing
def perform_hypothesis_testing(df, num_trials):
    mean_rec_loss = df.loc['reconstruction loss', 'mean']
    std_rec_loss = df.loc['reconstruction loss', 'std']
    mean_tan_loss = df.loc['tangent angle loss', 'mean']
    std_tan_loss = df.loc['tangent angle loss', 'std']
    n = num_trials

    # Calculate t-statistic
    t_statistic = (mean_rec_loss - mean_tan_loss) / np.sqrt((std_rec_loss ** 2 / n) + (std_tan_loss ** 2 / n))

    # Calculate degrees of freedom using Welch-Satterthwaite equation
    df_welch = ((std_rec_loss ** 2 / n) + (std_tan_loss ** 2 / n)) ** 2 / (
        (std_rec_loss ** 2 / n) ** 2 / (n - 1) + (std_tan_loss ** 2 / n) ** 2 / (n - 1))

    # Compute one-tailed p-value
    p_value = 1 - spstats.t.cdf(t_statistic, df=df_welch)
    t_critical = spstats.t.ppf(1 - ALPHA, df_welch)

    logger.info(f"T-statistic: {t_statistic}")
    logger.info(f"Critical region: {t_critical}")
    logger.info(f"Degrees of Freedom: {df_welch}")
    logger.info(f"One-tailed P-value: {p_value}")

    if t_statistic > t_critical:
        logger.info("Reject the null hypothesis (H0: recon loss <= tangent loss)")
    else:
        logger.info("Fail to reject the null hypothesis (H0: recon loss <= tangent loss)")


# Function for computing statistics
def compute_surface_stats(all_results):
    surface_stats = {}
    for i, surface_trials in enumerate(all_results):
        stats = {
            key: {
                "mean": np.mean([trial[key] for trial in surface_trials]),
                "std": np.std([trial[key] for trial in surface_trials]),
                "median": np.median([trial[key] for trial in surface_trials]),
                "max": np.max([trial[key] for trial in surface_trials]),
            } for key in surface_trials[0].keys()
        }
        surface_stats[f"surface_{i + 1}"] = stats
    return surface_stats


# Function for training and evaluating AE on a surface
def train_and_evaluate(surface_func, num_trials):
    trial_results = []
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])
    fuv = surface_func(u, v, C1, C2)
    chart = sp.Matrix([u, v, fuv])
    manifold = RiemannianManifold(local_coordinates, chart)

    local_drift = sp.Matrix([0, 0])
    local_diffusion = sp.Matrix([[1, 0], [0, 1]])

    bounds = [(A, B), (A, B)]
    large_bounds = [(A - EPSILON, B + EPSILON), (A - EPSILON, B + EPSILON)]

    for _ in range(num_trials):
        # Generate point cloud for training
        cloud = PointCloud(manifold, bounds, local_drift, local_diffusion)
        x, _, mu, cov, local_x = cloud.generate(NUM_TRAIN_POINTS, seed=None)
        x, mu, cov, p, orthogcomp, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)

        # AE model and loss
        ae = AutoEncoder1(INPUT_DIM, LATENT_DIM, HIDDEN_LAYERS, ENCODER_ACT, DECODER_ACT)
        weights = LossWeights()
        ae_loss = TotalLoss(weights, "fro", 0.)

        # Training
        fit_model2(ae, ae_loss, x, targets=(p, orthonormal_frame, cov, mu), lr=LEARNING_RATE, epochs=NUM_EPOCHS,
                   batch_size=BATCH_SIZE)

        # Generate point cloud for testing
        cloud_test = PointCloud(manifold, large_bounds, local_drift, local_diffusion)
        x_test, _, mu_test, cov_test, _ = cloud_test.generate(NUM_TEST_POINTS, seed=None)
        x_test, mu_test, cov_test, p_test, orthogcomp_test, orthonormal_frame_test = process_data(
            x_test, mu_test, cov_test, d=2, return_frame=True)

        # Evaluate losses after training
        losses = compute_test_losses(ae, ae_loss, x_test, p_test, orthonormal_frame_test, cov_test, mu_test)
        trial_results.append(losses)

    return trial_results


# Main function
def main():
    surfaces = get_surfaces()
    all_results = []

    for surface_func in surfaces:
        results = train_and_evaluate(surface_func, NUM_TRIALS)
        all_results.append(results)

    # Compute statistics over all trials
    surface_stats = compute_surface_stats(all_results)

    # Create DataFrames for each surface and perform hypothesis testing
    dfs = []
    for surface, stats in surface_stats.items():
        df = pd.DataFrame(stats).T  # Transpose to have rows as loss types and columns as statistics
        dfs.append(df)

        logger.info(f"\nStatistics for {surface}:")
        logger.info(df)

        perform_hypothesis_testing(df, NUM_TRIALS)

        # Generate LaTeX code for each surface
        latex_table = df.to_latex(float_format="%.6f", index=True)
        logger.info(f"\nLaTeX table for {surface}:")
        logger.info(latex_table)

    # Calculate the average across all surfaces
    average_df = pd.concat(dfs).groupby(level=0).mean()

    logger.info("\nAverage Statistics across all surfaces:")
    logger.info(average_df)

    # Generate LaTeX for the average statistics
    average_latex_table = average_df.to_latex(float_format="%.6f", index=True)
    logger.info("\nLaTeX table for the average statistics across all surfaces:")
    logger.info(average_latex_table)


if __name__ == "__main__":
    main()

