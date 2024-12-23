from shillml.utils import fit_model, compute_test_losses
from shillml.models.autoencoders import AutoEncoder1
from shillml.models.nsdes import AutoEncoderDiffusion2
from shillml.models.nsdes import AutoEncoderDrift, LatentNeuralSDE
from shillml.losses.loss_modules import TotalLoss
from shillml.losses import DiffusionLoss3
from shillml.losses import LatentDriftMSE


def train_autoencoder(
    input_data, targets, input_dim, latent_dim, hidden_layers,
    encoder_act, decoder_act, weights, norm, rank_scale, lr, epochs,
    batch_size, weight_decay, normalize=False, verbose=True
):
    """
    Train an autoencoder model on given input data and targets.
    """
    # Define the model
    model = AutoEncoder1(input_dim, latent_dim, hidden_layers, encoder_act, decoder_act)
    loss_fn = TotalLoss(weights, norm, normalize=normalize, scale=rank_scale)

    # Pre-training losses
    if verbose:
        print("Pre-training losses:")
        mu_train, cov_train, p_train, _, frame_train = targets  # Unpack targets explicitly
        pre_train_losses = compute_test_losses(model, loss_fn, input_data, p_train, frame_train, cov_train, mu_train)
        for key, value in pre_train_losses.items():
            print(f"  {key}: {value:.4f}")

    # Train the model
    fit_model(
        model=model,
        loss=loss_fn,
        input_data=input_data,
        targets=targets,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay
    )

    if verbose:
        print("\nAutoencoder training completed.")
    return model



def train_diffusion(
        ae, input_data, targets, latent_dim, hidden_layers,
        drift_act, diffusion_act, latent_cov_weight, ambient_cov_weight,
        diffusion_tangent_drift_weight, norm, normalize, lr, epochs,
        batch_size, verbose=True
):
    """
    Train the diffusion model with the autoencoder backbone.
    """
    # Define the diffusion model
    latent_sde = LatentNeuralSDE(latent_dim, hidden_layers, hidden_layers, drift_act, diffusion_act, None)
    model_diffusion = AutoEncoderDiffusion2(latent_sde, ae)

    # Define the diffusion loss
    diffusion_loss = DiffusionLoss3(
        latent_cov_weight=latent_cov_weight,
        ambient_cov_weight=ambient_cov_weight,
        tangent_drift_weight=diffusion_tangent_drift_weight,
        norm=norm,
        normalize=normalize
    )

    # Train the model
    if verbose:
        print("\nTraining diffusion model...")

    fit_model(
        model=model_diffusion,
        loss=diffusion_loss,
        input_data=input_data,
        targets=targets,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size
    )

    if verbose:
        print("\nDiffusion model training completed.")
    return model_diffusion


def train_drift(
        ae, latent_sde, input_data, targets, drift_lr, drift_epochs,
        batch_size, verbose=True
):
    """
    Train the drift model with the autoencoder backbone.
    """
    # Define the drift model
    model_drift = AutoEncoderDrift(latent_sde, ae)
    drift_loss = LatentDriftMSE()

    # Train the drift model
    if verbose:
        print("\nTraining drift model...")

    fit_model(
        model=model_drift,
        loss=drift_loss,
        input_data=input_data,
        targets=targets,
        lr=drift_lr,
        epochs=drift_epochs,
        batch_size=batch_size
    )

    if verbose:
        print("\nDrift model training completed.")
    return model_drift


def train_all_stages(
        input_data, ae_targets, diffusion_targets, drift_targets,
        input_dim, latent_dim, hidden_layers,
        encoder_act, decoder_act, drift_act, diffusion_act,
        weights, norm, rank_scale, lr, drift_lr, epochs_ae, epochs_diffusion, epochs_drift,
        batch_size, weight_decay, latent_cov_weight, ambient_cov_weight,
        diffusion_tangent_drift_weight, normalize=False, verbose=True
):
    """
    Train the full three-stage model: Autoencoder, Diffusion, and Drift.

    Args:
        input_data (torch.Tensor): Input data for training.
        ae_targets (tuple): Targets for autoencoder training (p, orthonormal_frame, cov, mu).
        diffusion_targets (tuple): Targets for diffusion training (mu, cov, encoded_cov, orthonormal_frame).
        drift_targets (tuple): Targets for drift training (mu, encoded_cov).
        input_dim (int): Input dimensionality.
        latent_dim (int): Latent dimensionality.
        hidden_layers (list): List of hidden layer sizes.
        encoder_act (nn.Module): Activation function for encoder.
        decoder_act (nn.Module): Activation function for decoder.
        drift_act (nn.Module): Activation function for drift model.
        diffusion_act (nn.Module): Activation function for diffusion model.
        weights (LossWeights): Weights for loss computation.
        norm (str): Norm type for loss ("fro", etc.).
        rank_scale (float): Scaling factor for loss.
        lr (float): Learning rate for AE and diffusion stages.
        drift_lr (float): Learning rate for drift stage.
        epochs_ae (int): Number of epochs for AE training.
        epochs_diffusion (int): Number of epochs for diffusion training.
        epochs_drift (int): Number of epochs for drift training.
        batch_size (int): Batch size for training.
        weight_decay (float): Weight decay (L2 regularization).
        latent_cov_weight (float): Weight for latent covariance penalty in diffusion.
        ambient_cov_weight (float): Weight for ambient covariance penalty in diffusion.
        diffusion_tangent_drift_weight (float): Weight for diffusion tangent drift penalty.
        normalize (bool): Whether to normalize losses.
        verbose (bool): Whether to log progress.

    Returns:
        tuple: Trained autoencoder, diffusion, and drift models.
    """
    # Stage 1: Train Autoencoder
    if verbose:
        print("\n=== Stage 1: Training Autoencoder ===")
    ae_model = train_autoencoder(
        input_data=input_data,
        targets=ae_targets,
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
        verbose=verbose
    )

    # Stage 2: Train Diffusion Model
    if verbose:
        print("\n=== Stage 2: Training Diffusion Model ===")
    diffusion_model = train_diffusion(
        ae=ae_model,
        input_data=input_data,
        targets=diffusion_targets,
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
        verbose=verbose
    )

    # Stage 3: Train Drift Model
    if verbose:
        print("\n=== Stage 3: Training Drift Model ===")
    drift_model = train_drift(
        ae=ae_model,
        latent_sde=diffusion_model.latent_sde,
        input_data=input_data,
        targets=drift_targets,
        drift_lr=drift_lr,
        drift_epochs=epochs_drift,
        batch_size=batch_size,
        verbose=verbose
    )

    # Return trained models
    if verbose:
        print("\n=== Training Completed ===")
    return ae_model, diffusion_model, drift_model
