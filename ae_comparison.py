# Experiment to test where contractive/tangent bundle AE beats vanilla AE
import os
import sympy as sp
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from shillml.diffgeo import RiemannianManifold
from shillml.models.autoencoders import AE, CAE, CHAE, TBAE, CTBAE, DACTBAE
from shillml.losses import AELoss, CAELoss, CHAELoss, TBAELoss, CTBAELoss, DACTBAELoss
from shillml.pointclouds import PointCloud, SDECoefficients, Surfaces
from shillml.utils import process_data, fit_model

# see shillml.point_clouds.surfaces and .dynamics for more options
surface = "paraboloid"  # choices: paraboloid, quartic, sphere, torus, gaussian_bump, hyperboloid, etc
dynamics = "langevin harmonic"  # choices: bm, rbm, langevin harmonic, langevin double well, arbitrary
num_pts = 30
batch_size = 5
num_test = 100
seed = 17
bd_epsilon = 0.5

# Encoder region and quiver length
alpha = -1
beta = 1
# Regularization: contract and tangent
contractive_weight = 0.0001
second_order_weight = 0.0001
tangent_bundle_weight = 0.001  # This is the weight for |P_model-P_true|_F^2
tangent_drift_weight = 0.001  # This is the weight for the |N(mu-0.5 q)|_2^2 drift alignment on Stage 1
tangent_bundle_norm = "fro"  # This for switching that above matrix norm in stage 1 above for the P's
lr = 0.0001
epochs = 20000
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
        manifold.local_bm_drift() + 0.5 * manifold.metric_tensor().inv() * coefs.drift_double_well_potential(),
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
x, mu, cov, P, PNN = process_data(x, mu, cov, d=intrinsic_dim)
# point_cloud.plot_sample_paths()
# point_cloud.plot_drift_vector_field(x, None, mu, 0.1)

# Interpolation testing point cloud
x_interp, _, mu_interp, cov_interp, _ = point_cloud.generate(n=num_test, seed=None)
x_interp, mu_interp, cov_interp, P_interp, N_interp = process_data(x_interp, mu_interp, cov_interp, d=intrinsic_dim)

# Extrapolation testing point cloud
point_cloud = PointCloud(manifold, bounds=large_bounds, local_drift=true_drift, local_diffusion=true_diffusion)
x_extrap, _, mu_extrap, cov_extrap, _ = point_cloud.generate(n=num_test, seed=None)
x_extrap, mu_extrap, cov_extrap, P_extrap, N_extrap = process_data(x_extrap, mu_extrap, cov_extrap, d=intrinsic_dim)

# Define models
ae = AE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
cae = CAE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
c2ae = CHAE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
tbae = TBAE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
ctbae = CTBAE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
curve_ae = DACTBAE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)

# Define the loss functions for each auto encoder
ae_loss = AELoss()
cae_loss = CAELoss(contractive_weight=contractive_weight)
c2ae_loss = CHAELoss(contractive_weight=contractive_weight, hessian_weight=second_order_weight)
tbae_loss = TBAELoss(tangent_bundle_weight=tangent_bundle_weight, norm=tangent_bundle_norm)
ctbae_loss = CTBAELoss(contractive_weight=contractive_weight, tangent_bundle_weight=tangent_bundle_weight,
                       norm=tangent_bundle_norm)
curve_loss = DACTBAELoss(contractive_weight=contractive_weight, tangent_drift_weight=tangent_drift_weight,
                         tangent_bundle_weight=tangent_bundle_weight, norm=tangent_bundle_norm)

# Run the program.
if __name__ == "__main__":
    fit_model(ae, ae_loss, x, None, lr, epochs, print_freq, weight_decay, batch_size)
    fit_model(cae, cae_loss, x, None, lr, epochs, print_freq, weight_decay, batch_size)
    fit_model(c2ae, c2ae_loss, x, None, lr, epochs, print_freq, weight_decay, batch_size)
    fit_model(tbae, tbae_loss, x, P, lr, epochs, print_freq, weight_decay, batch_size)
    fit_model(ctbae, ctbae_loss, x, P, lr, epochs, print_freq, weight_decay, batch_size)
    fit_model(curve_ae, curve_loss, x, (P, PNN, mu, cov), lr, epochs, print_freq, weight_decay, batch_size)
    # This computes the total loss of each model, which is probably an unfair comparison, no?
    # interpolation_error_ae = ae_loss.forward(ae, x_interp).item()
    # interpolation_error_cae = cae_loss.forward(cae, x_interp)
    # interpolation_error_c2ae = c2ae_loss.forward(c2ae, x_interp)
    # interpolation_error_tbae = tbae_loss.forward(tbae, x_interp, P_interp)
    # interpolation_error_ctbae = ctbae_loss.forward(ctbae, x_interp, P_interp)
    # interpolation_error_curve_ae = curve_loss.forward(curve_ae, x_interp, (P_interp, N_interp, mu_interp, cov_interp))
    # # Compute extrapolation error
    # extrapolation_error_ae = ae_loss.forward(ae, x_extrap).item()
    # extrapolation_error_cae = cae_loss.forward(cae, x_extrap)
    # extrapolation_error_c2ae = c2ae_loss.forward(c2ae, x_extrap)
    # extrapolation_error_tbae = tbae_loss.forward(tbae, x_extrap, P_extrap)
    # extrapolation_error_ctbae = ctbae_loss.forward(ctbae, x_extrap, P_extrap)
    # extrapolation_error_curve_ae = curve_loss.forward(curve_ae, x_extrap, (P_extrap, N_extrap, mu_extrap, cov_extrap))

    # Instead lets just compute the reconstruction error of each model, the easiest way to do this
    # is to re-declare the loss functions with zero weights:
    ae_loss = AELoss()
    cae_loss = CAELoss(contractive_weight=0.)
    c2ae_loss = CHAELoss(contractive_weight=0., hessian_weight=0.)
    tbae_loss = TBAELoss(tangent_bundle_weight=0., norm=tangent_bundle_norm)
    ctbae_loss = CTBAELoss(contractive_weight=0., tangent_bundle_weight=0.,
                           norm=tangent_bundle_norm)
    curve_loss = DACTBAELoss(contractive_weight=0., tangent_drift_weight=0.,
                             tangent_bundle_weight=0., norm=tangent_bundle_norm)

    interpolation_error_ae = ae_loss.forward(ae, x_interp).item()
    interpolation_error_cae = cae_loss.forward(cae, x_interp)
    interpolation_error_c2ae = c2ae_loss.forward(c2ae, x_interp)
    interpolation_error_tbae = tbae_loss.forward(tbae, x_interp, P_interp)
    interpolation_error_ctbae = ctbae_loss.forward(ctbae, x_interp, P_interp)
    interpolation_error_curve_ae = curve_loss.forward(curve_ae, x_interp, (P_interp, N_interp, mu_interp, cov_interp))
    # Compute extrapolation error
    extrapolation_error_ae = ae_loss.forward(ae, x_extrap).item()
    extrapolation_error_cae = cae_loss.forward(cae, x_extrap)
    extrapolation_error_c2ae = c2ae_loss.forward(c2ae, x_extrap)
    extrapolation_error_tbae = tbae_loss.forward(tbae, x_extrap, P_extrap)
    extrapolation_error_ctbae = ctbae_loss.forward(ctbae, x_extrap, P_extrap)
    extrapolation_error_curve_ae = curve_loss.forward(curve_ae, x_extrap, (P_extrap, N_extrap, mu_extrap, cov_extrap))

    # diffeomorphism error
    interp_diffeo_error = [model.compute_diffeo_error(x_interp).detach() for model in
                           [ae, cae, c2ae, tbae, ctbae, curve_ae]]
    extrap_diffeo_error = [model.compute_diffeo_error(x_extrap).detach() for model in
                           [ae, cae, c2ae, tbae, ctbae, curve_ae]]
    print("Interpolation diffeo error")
    print(interp_diffeo_error)
    print("Extrapolation diffeo error")
    print(extrap_diffeo_error)
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
        for error_type, error_value in error_types.items():
            print(f"{model} {error_type} Error: {error_value:.6f}")

    # Generate LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
    latex_table += "Model & Interpolation Error & Extrapolation Error \\\\ \\hline\n"

    for model, error_types in errors.items():
        latex_table += f"{model} & {error_types['Interpolation']:.6f} & {error_types['Extrapolation']:.6f} \\\\ \\hline\n"

    latex_table += "\\end{tabular}\n"

    # Create a detailed caption
    caption = f"Interpolation and Extrapolation Errors for Different Autoencoder Models. "
    caption += f"Surface: {surface}. "
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
    latex_table += "\\label{tab:ae_errors}\n\\end{table}"

    # Print LaTeX table to console
    print("\nLaTeX Table:")
    print(latex_table)
    # Create the directory structure
    os.makedirs(f"plots/{surface}/autoencoder", exist_ok=True)
    # Optionally, save LaTeX table to a file
    with open(f"plots/{surface}/autoencoder/error_table.tex", "w") as f:
        f.write(latex_table)
    torch.save(curve_ae.state_dict(), f"plots/{surface}/autoencoder/curve_ae.pth")

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
    c2ae.plot_surface(alpha, beta, 30, ax, "C2AE")

    ax = fig.add_subplot(2, 3, 4, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    tbae.plot_surface(alpha, beta, 30, ax, "TBAE")

    ax = fig.add_subplot(2, 3, 5, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    ctbae.plot_surface(alpha, beta, 30, ax, "CTBAE")

    ax = fig.add_subplot(2, 3, 6, projection="3d")
    ax.scatter3D(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2])
    curve_ae.plot_surface(alpha, beta, 30, ax, "DCTBAE")
    plt.show()
