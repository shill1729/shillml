"""
Run this BEFORE 'sde_comparison.py'.

This script tests the performance of various autoencoder models (AE, CAE, CHAE, TBAE, CTBAE, DACTBAE)
on a point cloud generated from a Riemannian manifold with specified surface and dynamics. It compares
interpolation and extrapolation errors across models, calculates diffeomorphism errors, and generates a
LaTeX table summarizing the results. The models are trained with different loss functions incorporating
contractive, second-order, and tangent bundle regularizations, and their reconstruction performance is
evaluated and visualized.

"""
# Experiment to test where contractive/tangent bundle AE beats vanilla AE
import os
import sympy as sp
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from shillml.diffgeo import RiemannianManifold
from shillml.models.autoencoders import AutoEncoder1
from shillml.losses.loss_modules import TotalLoss
from shillml.pointclouds import PointCloud, SDECoefficients, Surfaces
from shillml.utils import process_data, fit_model, compute_test_losses

# see shillml.point_clouds.surfaces and .dynamics for more options
surface = "paraboloid"  # choices: paraboloid, quartic, sphere, torus, gaussian_bump, hyperboloid, etc
dynamics = "rbm"  # choices: bm, rbm, langevin harmonic, langevin double well, arbitrary
num_pts = 30
batch_size = 30
num_test = 100
seed = 17
bd_epsilon = 0.5

# Encoder region and quiver length
alpha = -1
beta = 1
# Regularization: contract and tangent
contractive_weight = 0.
second_order_weight = 0.
tangent_bundle_weight = 1.  # This is the weight for |P_model-P_true|_F^2
tangent_drift_weight = 0.  # This is the weight for the |N(mu-0.5 q)|_2^2 drift alignment on Stage 1
tangent_bundle_norm = "fro"  # This for switching that above matrix norm in stage 1 above for the P's
lr = 0.001
epochs = 15000
print_freq = 1000
weight_decay = 0.
# Network structure
extrinsic_dim = 3
intrinsic_dim = 2
h1 = [64]
encoder_act = nn.Tanh()
decoder_act = nn.Tanh()
if seed is not None:
    torch.manual_seed(seed)

surfaces = Surfaces()
u, v = sp.symbols("u v", real=True)
local_coordinates = sp.Matrix([u, v])
# Assume that 'surface' is the user's input
if surface in surfaces.surface_bounds:
    # Set the chart dynamically
    chart = getattr(surfaces, surface)()
    # Set the bounds from the dictionary
    bounds = surfaces.surface_bounds[surface]
    # Set large_bounds (if needed)
    large_bounds = [(b[0] - bd_epsilon, b[1] + bd_epsilon) for b in bounds]
else:
    raise ValueError("Invalid surface")

coefs = SDECoefficients()
# Initialize the manifold and choose the dynamics
manifold = RiemannianManifold(local_coordinates, chart)
# Mapping of user input to predefined dynamics
dynamics_map = {
    "bm": (coefs.drift_zero(), coefs.diffusion_identity()),
    "rbm": (manifold.local_bm_drift(), manifold.local_bm_diffusion()),
    "langevin harmonic": (
        manifold.local_bm_drift() + 0.5 * manifold.metric_tensor().inv() * coefs.drift_harmonic_potential(),
        manifold.local_bm_diffusion()
    ),
    "langevin double well": (
        manifold.local_bm_drift() + 0.1 * manifold.metric_tensor().inv() * coefs.drift_double_well_potential(),
        manifold.local_bm_diffusion()
    ),
    "arbitrary": (
        manifold.local_bm_drift() + manifold.metric_tensor().inv() * coefs.drift_double_well_potential() * 0.25,
        manifold.local_bm_diffusion() * coefs.diffusion_circular() * 0.25
    )
}
true_drift, true_diffusion = dynamics_map[dynamics]

# Initialize the point cloud
point_cloud = PointCloud(manifold, bounds=bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x, w, mu, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
x, mu, cov, P, PNN, H = process_data(x, mu, cov, d=intrinsic_dim, return_frame=True)
# point_cloud.plot_sample_paths()
# point_cloud.plot_drift_vector_field(x, None, mu, 0.1)

# Interpolation testing point cloud
x_interp, _, mu_interp, cov_interp, _ = point_cloud.generate(n=num_test, seed=None)
x_interp, mu_interp, cov_interp, P_interp, N_interp, H_interp = process_data(x_interp, mu_interp, cov_interp,
                                                                             d=intrinsic_dim, return_frame=True)

# Extrapolation testing point cloud
point_cloud = PointCloud(manifold, bounds=large_bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x_extrap, _, mu_extrap, cov_extrap, _ = point_cloud.generate(n=num_test, seed=None)
x_extrap, mu_extrap, cov_extrap, P_extrap, N_extrap, H_extrap = process_data(x_extrap, mu_extrap, cov_extrap,
                                                                             d=intrinsic_dim, return_frame=True)

# Define models
ae = AutoEncoder1(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
cae = AutoEncoder1(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
frae = AutoEncoder1(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
tbae = AutoEncoder1(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
ctbae = AutoEncoder1(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
dactbae = AutoEncoder1(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)

# Define the loss functions for each auto encoder
ae_weights = {"rank_penalty": 0.,
              "contractive_reg": 0.,
              "tangent_bundle": 0.,
              "drift_alignment": 0.,
              "diffeo_reg1": 0.,
              "diffeo_reg2": 0.}
cae_weights = {"rank_penalty": 0.,
               "contractive_reg": contractive_weight,
               "tangent_bundle": 0.,
               "drift_alignment": 0.,
               "diffeo_reg1": 0.,
               "diffeo_reg2": 0.}
fr_weights = {"rank_penalty": 0.001,
              "contractive_reg": 0.,
              "tangent_bundle": 0.,
              "drift_alignment": 0.,
              "diffeo_reg1": 0.,
              "diffeo_reg2": 0.}
tbae_weights = {"rank_penalty": 0.,
              "contractive_reg": 0.,
              "tangent_bundle": tangent_bundle_weight,
              "drift_alignment": 0.,
              "diffeo_reg1": 0.,
              "diffeo_reg2": 0.}
ctbae_weights = {"rank_penalty": 0.,
              "contractive_reg": contractive_weight,
              "tangent_bundle": tangent_bundle_weight,
              "drift_alignment": 0.,
              "diffeo_reg1": 0.,
              "diffeo_reg2": 0.}
dactbae_weights = {"rank_penalty": 0.,
              "contractive_reg": contractive_weight,
              "tangent_bundle": tangent_bundle_weight,
              "drift_alignment": tangent_drift_weight,
              "diffeo_reg1": 0.,
              "diffeo_reg2": 0.}

ae_loss = TotalLoss(ae_weights)
cae_loss = TotalLoss(cae_weights)
frae_loss = TotalLoss(fr_weights)
tbae_loss = TotalLoss(tbae_weights) # TODO add norm optionality
ctbae_loss = TotalLoss(ctbae_weights)
dactbae_loss = TotalLoss(dactbae_weights)

# Run the program.
if __name__ == "__main__":
    print("Training Auto encoder")
    fit_model(ae, ae_loss, x, (P, H, cov, mu), lr, epochs, print_freq, weight_decay, batch_size)
    print("Training contractive AE")
    fit_model(cae, cae_loss, x, (P, H, cov, mu), lr, epochs, print_freq, weight_decay, batch_size)
    print("Training Full-Rank AE")
    fit_model(frae, frae_loss, x, (P, H, cov, mu), lr, epochs, print_freq, weight_decay, batch_size)
    print("Training Tangent-Bundle regularized AE")
    fit_model(tbae, tbae_loss, x, (P, H, cov, mu), lr, epochs, print_freq, weight_decay, batch_size)
    print("Training Tangent Bundle CAE")
    fit_model(ctbae, ctbae_loss, x, (P, H, cov, mu), lr, epochs, print_freq, weight_decay, batch_size)
    print("Training Drift Aligned TBCAE")
    fit_model(dactbae, dactbae_loss, x, (P, H, cov, mu), lr, epochs, print_freq, weight_decay, batch_size)

    # Compute interpolation error
    interpolation_error_ae = compute_test_losses(ae, ae_loss, x_interp, P_interp, H_interp, cov_interp, mu_interp)
    interpolation_error_cae = compute_test_losses(cae, cae_loss, x_interp, P_interp, H_interp, cov_interp, mu_interp)
    interpolation_error_c2ae = compute_test_losses(frae, frae_loss, x_interp, P_interp, H_interp, cov_interp, mu_interp)
    interpolation_error_tbae = compute_test_losses(tbae, tbae_loss, x_interp, P_interp, H_interp, cov_interp, mu_interp)
    interpolation_error_ctbae = compute_test_losses(ctbae, ctbae_loss, x_interp, P_interp, H_interp, cov_interp, mu_interp)
    interpolation_error_curve_ae = compute_test_losses(dactbae, dactbae_loss, x_interp, P_interp, H_interp, cov_interp, mu_interp)
    # Compute extrapolation error
    extrapolation_error_ae = compute_test_losses(ae, ae_loss, x_extrap, P_extrap, H_extrap, cov_extrap, mu_extrap)
    extrapolation_error_cae = compute_test_losses(cae, cae_loss, x_extrap, P_extrap, H_extrap, cov_extrap, mu_extrap)
    extrapolation_error_c2ae = compute_test_losses(frae, frae_loss, x_extrap, P_extrap, H_extrap, cov_extrap, mu_extrap)
    extrapolation_error_tbae = compute_test_losses(tbae, tbae_loss, x_extrap, P_extrap, H_extrap, cov_extrap, mu_extrap)
    extrapolation_error_ctbae = compute_test_losses(ctbae, ctbae_loss, x_extrap, P_extrap, H_extrap, cov_extrap, mu_extrap)
    extrapolation_error_curve_ae = compute_test_losses(dactbae, dactbae_loss, x_extrap, P_extrap, H_extrap, cov_extrap, mu_extrap)

    # print("Interpolation diffeo error")
    # print(interp_diffeo_error)
    # print("Extrapolation diffeo error")
    # print(extrap_diffeo_error)
    # After calculating all errors
    # Create a dictionary to store all errors
    errors = {
        "AE": {"Interpolation": interpolation_error_ae, "Extrapolation": extrapolation_error_ae},
        "CAE": {"Interpolation": interpolation_error_cae, "Extrapolation": extrapolation_error_cae},
        "C2AE": {"Interpolation": interpolation_error_c2ae, "Extrapolation": extrapolation_error_c2ae},
        "TBAE": {"Interpolation": interpolation_error_tbae, "Extrapolation": extrapolation_error_tbae},
        "CTBAE": {"Interpolation": interpolation_error_ctbae, "Extrapolation": extrapolation_error_ctbae},
        "DACTBAE": {"Interpolation": interpolation_error_curve_ae, "Extrapolation": extrapolation_error_curve_ae}
    }

    # Print errors to console
    print("\nErrors:")
    for model, error_types in errors.items():
        for error_type, error_dict in error_types.items():
            print(f"{model} {error_type} Errors:")
            for loss_name, loss_value in error_dict.items():
                print(f"  {loss_name}: {loss_value:.6f}")

    # Generate LaTeX table with full caption and detailed information
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n\\hline\n"
    latex_table += "Model &  Recon. &  Contract & Rank  & Tangent  & Drift  & Diffeo 1 & Diffeo 2 \\\\ \\hline\n"

    for model, error_types in errors.items():
        interp_losses = error_types['Interpolation']
        latex_table += f"{model} & {interp_losses['reconstruction_loss']:.6f} & {interp_losses['encoder_contractive_loss']:.6f} & "
        latex_table += f"{interp_losses['rank_penalty']:.6f} & {interp_losses['tangent_bundle_loss']:.6f} & "
        latex_table += f"{interp_losses['drift_alignment_loss']:.6f} & {interp_losses['diffeomorphism_loss1']:.6f} & {interp_losses['diffeomorphism_loss2']:.6f} \\\\ \\hline\n"

    latex_table += "\\end{tabular}\n"

    # Generate LaTeX table with full caption and detailed information
    latex_table2 = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n\\hline\n"
    latex_table2 += "Model & Recon. & Contract & Rank & Tangent & Drift & Diffeo 1 & Diffeo 2 \\\\ \\hline\n"

    for model, error_types in errors.items():
        extrap_losses = error_types['Extrapolation']
        latex_table2 += f"{model} & {extrap_losses['reconstruction_loss']:.6f} & {extrap_losses['contractive_loss']:.6f} & "
        latex_table2 += f"{extrap_losses['rank_penalty']:.6f} & {extrap_losses['tangent_bundle_loss']:.6f} & "
        latex_table2 += f"{extrap_losses['drift_alignment_loss']:.6f} & {extrap_losses['diffeomorphism_loss1']:.6f} & {extrap_losses['diffeomorphism_loss2']:.6f} \\\\ \\hline\n"

    latex_table2 += "\\end{tabular}\n"

    # Detailed caption that includes all your parameters
    caption = f"Interpolation and Extrapolation Errors for Different Autoencoder Models. "
    caption += f"Surface: {surface}. "
    caption += f"Local dynamics: {dynamics}. "
    caption += f"Network dimensions: {extrinsic_dim} (extrinsic) to {intrinsic_dim} (intrinsic). "
    caption += f"Hidden layers: {h1}. "
    caption += f"Encoder activation: {encoder_act.__class__.__name__}. "
    caption += f"Decoder activation: {decoder_act.__class__.__name__}. "
    caption += f"Training epochs: {epochs}. "
    caption += f"Learning rate: {lr}. "
    caption += f"Weight decay: {weight_decay}. "
    caption += f"Contractive weight: {contractive_weight}. "
    caption += f"2nd-order Contraction weight: {second_order_weight}. "
    caption += f"Tangent bundle weight: {tangent_bundle_weight}. "
    caption += f"Tangent drift weight: {tangent_drift_weight}. "
    caption += f"Training region bounds: {bounds}. "
    caption += f"Extrapolation region bounds: {large_bounds}. "
    caption += f"Number of training points: {num_pts}. "
    caption += f"Number of test points: {num_test}. "
    caption += f"Random seed: {seed}."

    latex_table += f"\\caption{{{caption}}}\n"
    latex_table += "\\label{tab:ae_errors_interp}\n\\end{table}"

    latex_table2 += f"\\caption{{{caption}}}\n"
    latex_table2 += "\\label{tab:ae_errors_extrap}\n\\end{table}"

    # Print LaTeX table to console
    print("\nLaTeX Table:")
    print(latex_table)

    # Print LaTeX table to console
    print("\nLaTeX Table:")
    print(latex_table2)

    # Optionally, save LaTeX table to a file
    os.makedirs(f"plots/{surface}/autoencoder", exist_ok=True)
    with open(f"plots/{surface}/autoencoder/error_table.tex", "w") as f:
        f.write(latex_table)
    with open(f"plots/{surface}/autoencoder/error_table_extrap.tex", "w") as f:
        f.write(latex_table2)

    # Save model state
    torch.save(dactbae.state_dict(), f"plots/{surface}/autoencoder/dactbae.pth")

    # Detach for plots!
    x = x.detach()
    x_extrap = x_extrap.detach()
    # Comparing 6 AEs
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    ae.plot_surface(alpha, beta, 30, ax, "AE")

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    cae.plot_surface(alpha, beta, 30, ax, "CAE")

    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    frae.plot_surface(alpha, beta, 30, ax, "C2AE")

    ax = fig.add_subplot(2, 3, 4, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    tbae.plot_surface(alpha, beta, 30, ax, "TBAE")

    ax = fig.add_subplot(2, 3, 5, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    ctbae.plot_surface(alpha, beta, 30, ax, "CTBAE")

    ax = fig.add_subplot(2, 3, 6, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    dactbae.plot_surface(alpha, beta, 30, ax, "DCTBAE")
    plt.show()
