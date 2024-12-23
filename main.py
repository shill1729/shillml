import torch
import torch.nn as nn
import logging
from shillml.losses.loss_modules import LossWeights
from shillml.utils import compute_test_losses, set_grad_tracking, process_data
from data_generation import define_manifold, generate_point_cloud
from training import train_drift, train_autoencoder, train_diffusion

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # === Experiment Configuration ===
    logger.info("Initializing configuration...")

    # General settings
    train_seed = 17
    test_seed = 42
    bounds = [(-1, 1), (-1, 1)]
    large_bounds = [(-1.05, 1.05), (-1.05, 1.05)]
    num_points = 30
    num_test_points = 100

    # Model and training settings
    input_dim, latent_dim = 3, 2
    hidden_layers = [16]
    epochs_ae, epochs_diffusion, epochs_drift = 10000, 10000, 20000
    batch_size = num_points
    lr, drift_lr = 0.01, 0.01
    weight_decay = 0.0
    normalize = False
    norm = "fro"
    rank_scale = 1.0

    # Loss weights
    weights = LossWeights()
    weights.encoder_contraction_weight = 0.0
    weights.decoder_contraction_weight = 0.0
    weights.tangent_angle_weight = 0.0
    weights.tangent_drift_weight = 0.0
    weights.diffeomorphism_reg1 = 0.0
    latent_cov_weight = 1.0
    ambient_cov_weight = 0.0
    diffusion_tangent_drift_weight = 0.0

    # Activation functions
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()
    drift_act = nn.Tanh()
    diffusion_act = nn.Tanh()

    # === Data Generation ===
    logger.info("Generating training data...")
    manifold = define_manifold(surface_type="paraboloid", c1=3, c2=3)
    train_data = generate_point_cloud(
        manifold=manifold,
        bounds=bounds,
        drift_type="arbitrary",
        num_points=num_points,
        seed=train_seed,
    )
    x_train, mu_train, cov_train, local_coords_train = train_data
    ae_targets_train = process_data(x_train, mu_train, cov_train, d=2, return_frame=True)
    x_train = ae_targets_train[0]
    ae_targets_train = ae_targets_train[1:][::-1]


    logger.info("Generating testing data...")
    test_data = generate_point_cloud(
        manifold=manifold,
        bounds=large_bounds,
        drift_type="arbitrary",
        num_points=num_test_points,
        seed=test_seed,
    )
    x_test, mu_test, cov_test, local_coords_test = test_data
    ae_targets_test = process_data(x_test, mu_test, cov_test, d=2, return_frame=True)
    x_test = ae_targets_test[0]
    ae_targets_test = ae_targets_test[1:]

    # === Stage 1: Train Autoencoder ===
    logger.info("Training autoencoder...")
    ae_model = train_autoencoder(
        input_data=x_train,
        targets=ae_targets_train,
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        encoder_act=encoder_act,
        decoder_act=decoder_act,
        weights=weights,
        norm=norm,
        rank_scale=rank_scale,
        lr=lr,
        epochs=epochs_ae,
        batch_size=batch_size,
        weight_decay=weight_decay,
        normalize=normalize,
        verbose=True,
    )
    set_grad_tracking(ae_model, False)

    # === Stage 2: Prepare Diffusion Targets and Train Diffusion Model ===
    logger.info("Preparing diffusion targets...")
    dpi_train = ae_model.encoder.jacobian_network(x_train).detach()
    encoded_cov_train = torch.bmm(torch.bmm(dpi_train, cov_train), dpi_train.mT)
    diffusion_targets_train = (*ae_targets_train[1:], encoded_cov_train)

    dpi_test = ae_model.encoder.jacobian_network(x_test).detach()
    encoded_cov_test = torch.bmm(torch.bmm(dpi_test, cov_test), dpi_test.mT)
    diffusion_targets_test = (*ae_targets_test[1:], encoded_cov_test)

    logger.info("Training diffusion model...")
    diffusion_model = train_diffusion(
        ae=ae_model,
        input_data=x_train,
        targets=diffusion_targets_train,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        drift_act=drift_act,
        diffusion_act=diffusion_act,
        latent_cov_weight=latent_cov_weight,
        ambient_cov_weight=ambient_cov_weight,
        diffusion_tangent_drift_weight=diffusion_tangent_drift_weight,
        norm=norm,
        normalize=normalize,
        lr=lr,
        epochs=epochs_diffusion,
        batch_size=batch_size,
        verbose=True,
    )
    set_grad_tracking(diffusion_model.latent_sde.diffusion_net, False)

    # === Stage 3: Train Drift Model ===
    logger.info("Training drift model...")
    drift_targets_train = (mu_train, encoded_cov_train)
    drift_targets_test = (mu_test, encoded_cov_test)
    drift_model = train_drift(
        ae=ae_model,
        latent_sde=diffusion_model.latent_sde,
        input_data=x_train,
        targets=drift_targets_train,
        drift_lr=drift_lr,
        drift_epochs=epochs_drift,
        batch_size=batch_size,
        verbose=True,
    )

    # === Evaluation ===
    logger.info("Evaluating model on testing data...")
    logger.info("Autoencoder testing losses:")
    ae_test_losses = compute_test_losses(
        ae_model, ae_targets_test[0], x_test, *ae_targets_test[1:]
    )
    for key, value in ae_test_losses.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("Diffusion testing losses:")
    diffusion_loss_test = diffusion_model.loss(
        x_test, diffusion_targets_test
    ).detach().numpy()
    logger.info(f"  Diffusion loss: {diffusion_loss_test}")

    logger.info("Drift testing losses:")
    drift_loss_test = drift_model.loss(x_test, drift_targets_test).detach().numpy()
    logger.info(f"  Drift loss: {drift_loss_test}")

    logger.info("Training pipeline completed successfully.")


# === Run the Script ===
if __name__ == "__main__":
    main()
