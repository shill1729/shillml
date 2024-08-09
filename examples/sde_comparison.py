"""
Run this AFTER 'ae_comparison.py'.

This script loads a trained DACTBAE model, freezes its parameters, and fits two Stochastic Differential
Equation (SDE) models—a drift SDE and a diffusion SDE—using the latent space of the autoencoder. The script
computes various losses related to reconstruction, tangent bundle alignment, contraction, and SDE consistency
for both interpolation and extrapolation tests. Results are summarized in a LaTeX table, and the model's
performance is visualized by plotting the learned drift against the true drift in 3D.
"""
# Load in the CCTBAE
# Fit the SDEs
from examples.ae_comparison import *
from shillml.models.nsdes import LatentNeuralSDE, AutoEncoderDrift, AutoEncoderDiffusion
from shillml.losses import DiffusionLoss, DriftMSELoss, MatrixMSELoss, contractive_regularization, tangent_drift_loss
from shillml.utils import set_grad_tracking, fit_model

diffusion_epochs = 30000
drift_epochs = 50000
sde_drift_alignment_weight = 0.0001
h2 = [32]
h3 = [64]
# Sample paths:
tn = 1
npaths = 10
ntime = 8000

sde_matrix_norm = "fro"
drift_act = nn.GELU()
diffusion_act = nn.GELU()
final_coef_act = nn.Tanh()


def load_model_weights(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set the model to evaluation mode
    return model


# Usage
ctbae = DACTBAE(extrinsic_dim, intrinsic_dim, h1, encoder_act, decoder_act)
ctbae = load_model_weights(ctbae, f"plots/{surface}/autoencoder/curve_ae.pth")
# Switch surface in ae_comparison.py to load different trained AEs
# ctbae = load_model_weights(ctbae, f"results/{surface}/{dynamics}/curve_ae.pth")

# Make sure the AE is frozen
set_grad_tracking(ctbae, False)

# Define SDEs
latent_sde = LatentNeuralSDE(intrinsic_dim, h2, h3, drift_act, diffusion_act, final_coef_act)
model_diffusion = AutoEncoderDiffusion(latent_sde, ctbae)
diffusion_loss = DiffusionLoss(sde_drift_alignment_weight, norm=sde_matrix_norm)
# estimate the encoded covariance via the trained AE
dpi = ctbae.encoder.jacobian_network(x).detach()
encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
# Fit the diffusion model
fit_model(model_diffusion, diffusion_loss, x, (mu, cov, encoded_cov), lr, diffusion_epochs,
          print_freq, weight_decay)
# Fit drift
set_grad_tracking(model_diffusion.latent_sde.diffusion_net, False)
model_drift = AutoEncoderDrift(latent_sde, ctbae)
drift_loss = DriftMSELoss()
fit_model(model_drift, drift_loss, x, (mu, encoded_cov), lr, drift_epochs, print_freq, weight_decay)


# Define helper function for computing losses
def compute_losses(x, mu, cov, P, NNproj):
    # Compute in-bound test loss
    mse_loss = nn.MSELoss()
    tbloss = MatrixMSELoss()
    contraction = contractive_regularization
    ctbae_loss_test = DACTBAELoss(contractive_weight, tangent_drift_weight, tangent_bundle_weight)
    x_test_reconstructed, dpi, P_model, hessian_decoder = ctbae(x)
    encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT).detach()
    mse = mse_loss(x, x_test_reconstructed).item()
    tb_loss = tbloss.forward(P_model, P).item()
    contraction_value = contraction(dpi).item()
    ctbae_loss_test_value = ctbae_loss_test(ctbae, x, (P, NNproj, mu, cov)).item()
    covariance_loss = MatrixMSELoss(norm=sde_matrix_norm)
    normal_bundle_loss = tangent_drift_loss
    model_cov, N, q, bbt = model_diffusion.forward(x)
    tangent_vector = mu - 0.5 * q
    normal_proj_vector = torch.bmm(N, tangent_vector.unsqueeze(2))
    normal_bundle_loss_value = normal_bundle_loss(normal_proj_vector).item()
    cov_mse_loss = covariance_loss.forward(model_cov, cov).item()
    total_diffusion_loss = diffusion_loss.forward(model_diffusion, x,  (mu, cov, encoded_cov)).item()
    drift_mse = drift_loss.forward(model_drift, x, (mu, encoded_cov)).item()
    loss_tuple = (mse, tb_loss, contractive_weight * contraction_value, ctbae_loss_test_value, cov_mse_loss,
                  normal_bundle_loss_value, total_diffusion_loss, drift_mse)
    return loss_tuple


interp_loss = compute_losses(x_interp, mu_interp, cov_interp, P_interp, N_interp)
extrap_loss = compute_losses(x_extrap, mu_extrap, cov_extrap, P_extrap, N_extrap)
# Unpack the loss tuples
(mse_interp, tb_loss_interp, contraction_interp, ctbae_loss_interp,
 cov_mse_loss_interp, normal_bundle_loss_interp, total_diffusion_loss_interp,
 drift_mse_interp) = interp_loss

(mse_extrap, tb_loss_extrap, contraction_extrap, ctbae_loss_extrap,
 cov_mse_loss_extrap, normal_bundle_loss_extrap, total_diffusion_loss_extrap,
 drift_mse_extrap) = extrap_loss

# Create a dictionary to store all errors
errors = {
    "Reconstruction (MSE)": {"Interpolation": mse_interp, "Extrapolation": mse_extrap},
    "Tangent Bundle": {"Interpolation": tb_loss_interp, "Extrapolation": tb_loss_extrap},
    "Contraction": {"Interpolation": contraction_interp, "Extrapolation": contraction_extrap},
    "Total CTBAE": {"Interpolation": ctbae_loss_interp, "Extrapolation": ctbae_loss_extrap},
    "Covariance MSE": {"Interpolation": cov_mse_loss_interp, "Extrapolation": cov_mse_loss_extrap},
    "Tangent Drift Alignment Error": {"Interpolation": normal_bundle_loss_interp, "Extrapolation": normal_bundle_loss_extrap},
    "Total Diffusion": {"Interpolation": total_diffusion_loss_interp, "Extrapolation": total_diffusion_loss_extrap},
    "Drift MSE": {"Interpolation": drift_mse_interp, "Extrapolation": drift_mse_extrap}
}

# Generate LaTeX table
latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
latex_table += "Loss Type & Interpolation Error & Extrapolation Error \\\\ \\hline\n"

for loss_type, error_types in errors.items():
    latex_table += f"{loss_type} & {error_types['Interpolation']:.6f} & {error_types['Extrapolation']:.6f} \\\\ \\hline\n"

latex_table += "\\end{tabular}\n"

# Create a detailed caption
caption = f"Interpolation and Extrapolation Errors for DACTBAE+SDE Model. "
caption += f"Surface: {surface}. "
caption += f"Local dynamics: {dynamics}. "
caption += f"Network dimensions: {extrinsic_dim} (extrinsic) to {intrinsic_dim} (intrinsic). "
caption += f"DACTBAE hidden layers: {h1}. "
caption += f"Drift SDE hidden layers: {h2}. "
caption += f"Diffusion SDE hidden layers: {h3}. "
caption += f"DACTBAE encoder activation: {encoder_act.__class__.__name__}. "
caption += f"DACTBAE decoder activation: {decoder_act.__class__.__name__}. "
caption += f"Drift activation: {drift_act.__class__.__name__}. "
caption += f"Diffusion activation: {diffusion_act.__class__.__name__}. "
caption += f"CTBAE training epochs: {epochs}. "
caption += f"Diffusion training epochs: {diffusion_epochs}. "
caption += f"Drift training epochs: {drift_epochs}. "
caption += f"Learning rate: {lr}. "
caption += f"Weight decay: {weight_decay}. "
caption += f"Contractive weight: {contractive_weight}. "
caption += f"Tangent bundle weight: {tangent_bundle_weight}. "
caption += f"sde-tangent drift alignment weight: {sde_drift_alignment_weight}. "
caption += f"Training region bounds: {bounds}. "
caption += f"Extrapolation region bounds: {large_bounds}. "
caption += f"Number of training points: {num_pts}. "
caption += f"Number of test points: {num_test}. "
caption += f"Random seed: {seed}."

latex_table += f"\\caption{{{caption}}}\n"
latex_table += "\\label{tab:ctbae_sde_errors}\n\\end{table}"

# Print LaTeX table to console
print("\nLaTeX Table:")
print(latex_table)

# Optionally, save LaTeX table to a file
os.makedirs(f"plots/{surface}/ctbae_sde", exist_ok=True)
with open(f"plots/{surface}/ctbae_sde/error_table.tex", "w") as f:
    f.write(latex_table)

mu_model = model_drift.forward(x_extrap).detach()
x_extrap = x_extrap.detach()
mu_extrap = mu_extrap.detach()

fig = plt.figure()
ax = plt.subplot(111, projection="3d")
ctbae.plot_surface(alpha, beta, 30, ax, "Drift+Tangent Aligned CAE")
ax.quiver(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2], mu_extrap[:, 0], mu_extrap[:, 1], mu_extrap[:, 2],
          normalize=True, length=0.15, label="True drift")
ax.quiver(x_extrap[:, 0], x_extrap[:, 1], x_extrap[:, 2], mu_model[:, 0], mu_model[:, 1], mu_model[:, 2],
          normalize=True, length=0.15, color="red", alpha=0.5, label="Model drift")
ax.legend()
plt.show()
