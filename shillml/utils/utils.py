"""
    This module contains utility functions for PyTorch.

    Specifically, we have

    set_grad_tracking: turn on/off the parameters of a nn.Module
    select_device: choose the computational device: cpu or gpu (cuda, or mps)
    process_data: take the point cloud/dynamics data and estimate the orthogonal projection

"""
from typing import Union, List, Tuple, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from shillml.models.autoencoders import AutoEncoder1
from shillml.losses.loss_modules import TotalLoss


def process_data(x, mu, cov, d, return_frame=False):
    x = torch.tensor(x, dtype=torch.float32)
    mu = torch.tensor(mu, dtype=torch.float32)
    cov = torch.tensor(cov, dtype=torch.float32)
    left_singular_vectors = torch.linalg.svd(cov)[0]
    orthonormal_frame = left_singular_vectors[:, :, 0:d]
    observed_projection = torch.bmm(orthonormal_frame, orthonormal_frame.mT)
    n, D, _ = observed_projection.size()
    observed_normal_projection = torch.eye(D).expand(n, D, D) - observed_projection
    if return_frame:
        return x, mu, cov, observed_projection, observed_normal_projection, orthonormal_frame
    else:
        return x, mu, cov, observed_projection, observed_normal_projection


def fit_model2(model: nn.Module,
               loss: nn.Module,
               input_data: Tensor,
               targets: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None,
               lr: float = 0.001,
               epochs: int = 1000,
               print_freq: int = 1000,
               weight_decay: float = 0.,
               batch_size: int = None,
               scheduler_step_size: int = 100,
               gamma: float = 0.1) -> None:
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define a scheduler, e.g., StepLR reduces the learning rate by gamma every 'scheduler_step_size' epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)
    # CosineAnnealingLR is another option...
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_step_size)

    # If batch_size is None or larger than dataset, use the full dataset as one batch
    if batch_size is None or batch_size > len(input_data):
        batch_size = len(input_data)

    # Create TensorDataset and DataLoader
    if targets is None:
        dataset = TensorDataset(input_data)
    elif isinstance(targets, (list, tuple)):
        dataset = TensorDataset(input_data, *targets)
    else:
        dataset = TensorDataset(input_data, targets)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0.0  # Reset the epoch loss at the start of each epoch

        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0]
            extra_targets = batch[1:] if len(
                batch) > 1 else None  # Extract remaining tensors in the batch as extra targets
            extra_targets = extra_targets[0] if extra_targets and len(extra_targets) == 1 else extra_targets
            loss_value = loss(model, inputs, extra_targets)
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item()  # Accumulate batch loss into epoch loss

        # Step the scheduler after each epoch
        scheduler.step()

        # Print average loss for the epoch if print_freq is met
        if epoch % print_freq == 0:
            print(f'Epoch: {epoch}: Train-Loss: {epoch_loss / len(dataloader):.6f}')
            print(f'Current Learning Rate: {scheduler.get_last_lr()}')

    return None


def fit_model(model: nn.Module,
              loss: nn.Module,
              input_data: Tensor,
              targets: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None,
              lr: float = 0.001,
              epochs: int = 1000,
              print_freq: int = 1000,
              weight_decay: float = 0.,
              batch_size: int = None) -> None:
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # If batch_size is None or larger than dataset, use the full dataset as one batch
    if batch_size is None or batch_size > len(input_data):
        batch_size = len(input_data)

    # Create TensorDataset and DataLoader
    if targets is None:
        dataset = TensorDataset(input_data)
    elif isinstance(targets, (list, tuple)):
        dataset = TensorDataset(input_data, *targets)
    else:
        dataset = TensorDataset(input_data, targets)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0.0  # Reset the epoch loss at the start of each epoch

        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0]
            extra_targets = batch[1:] if len(
                batch) > 1 else None  # Extract remaining tensors in the batch as extra targets
            extra_targets = extra_targets[0] if extra_targets and len(extra_targets) == 1 else extra_targets
            # if targets is None:
            #     loss_value = loss(model, inputs, None)
            # else:
            #     loss_value = loss(model, inputs, extra_targets)
            loss_value = loss(model, inputs, extra_targets)
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item()  # Accumulate batch loss into epoch loss

        # Print average loss for the epoch if print_freq is met
        if epoch % print_freq == 0:
            print(f'Epoch: {epoch}: Train-Loss: {epoch_loss / len(dataloader):.6f}')
    return None


def set_grad_tracking(model: nn.Module, enable: bool = False) -> None:
    """
    Enable or disable gradient tracking for a nn.Module's parameters.

    Parameters:
    model (nn.Module): The model for which to toggle gradient tracking.
    enable (bool, optional): True to enable gradient tracking, False to disable. Default is False.

    Returns:
    None
    """
    for parameter in model.parameters():
        parameter.requires_grad = enable
    return None


# define function to set device
def select_device(preferred_device=None):
    """
        Selects the appropriate device for PyTorch computations.

        Parameters:
        preferred_device (str, optional): The preferred device to use ('cuda', 'mps', or 'cpu').
                                          If not specified, the function will select the best available device.

        Returns:
        torch.device: The selected device.

        If the preferred device is not available, it falls back to the first available device in the order of
        'cuda', 'mps', 'cpu'.
    """
    available_devices = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True  # cpu is always available
    }

    if preferred_device:
        if preferred_device in available_devices and available_devices[preferred_device]:
            device = torch.device(preferred_device)
            print(f"Using {preferred_device.upper()}.")
        else:
            print(f"{preferred_device.upper()} is not available. Falling back to available devices.")
            device = next((torch.device(dev) for dev, available in available_devices.items() if available),
                          torch.device("cpu"))
            print(f"Using {device.type.upper()}.")
    else:
        device = next((torch.device(dev) for dev, available in available_devices.items() if available),
                      torch.device("cpu"))
        print(f"Using {device.type.upper()}.")

    return device


def compute_test_losses(ae: AutoEncoder1, ae_loss: TotalLoss, x_test, p_test, frame_test, cov_test, mu_test):
    n, D, _ = p_test.size()
    normal_proj_test = torch.eye(D).expand(n, D, D) - p_test
    # Get reconstructed test data
    x_test_recon = ae.decoder(ae.encoder(x_test))

    # Compute test reconstruction error
    reconstruction_loss_test = ae_loss.reconstruction_loss.forward(x_test_recon, x_test).item()

    # Compute Jacobians and Hessians
    decoder_jacobian_test = ae.decoder_jacobian(ae.encoder(x_test))
    encoder_jacobian_test = ae.encoder_jacobian(x_test)
    decoder_hessian_test = ae.decoder_hessian(ae.encoder(x_test))

    # Contractive regularization
    contractive_loss_test = ae_loss.contractive_reg(encoder_jacobian_test).item()

    # Rank penalty
    rank_penalty_test = ae_loss.rank_penalty(decoder_jacobian_test).item()

    # Neural metric tensor
    metric_tensor_test = torch.bmm(decoder_jacobian_test.mT, decoder_jacobian_test)
    # Tangent error
    tangent_bundle_loss_test = ae_loss.tangent_bundle_reg.forward(decoder_jacobian_test,
                                                                  metric_tensor_test,
                                                                  p_test).item()
    # Drift alignment regularization
    drift_alignment_loss_test = ae_loss.drift_alignment_reg.forward(encoder_jacobian_test,
                                                                    decoder_hessian_test,
                                                                    cov_test,
                                                                    mu_test,
                                                                    normal_proj_test).item()

    # Diffeomorphism regularization 1
    diffeomorphism_loss1_test = ae_loss.diffeomorphism_reg1(decoder_jacobian_test, encoder_jacobian_test).item()

    variance_logdet_loss = ae_loss.variance_log_det_reg(metric_tensor_test).item()

    decoder_contraction = ae_loss.contractive_reg(decoder_jacobian_test).item()

    orthogonal_coordinates_error = ae_loss.orthogonal_coordinates(metric_tensor_test).item()

    tangent_angle_loss = ae_loss.tangent_angles_reg.forward(frame_test, decoder_jacobian_test, metric_tensor_test).item()

    normal_component_loss = ae_loss.normal_component_reg.forward(x_test, x_test_recon, frame_test).item()

    # Return all the losses in a dictionary
    return {
        "reconstruction loss": reconstruction_loss_test,
        "encoder contractive loss": contractive_loss_test,
        "decoder contractive loss": decoder_contraction,
        "rank penalty": rank_penalty_test,
        "tangent bundle loss": tangent_bundle_loss_test,
        "tangent angle loss": tangent_angle_loss,
        "normal component loss": normal_component_loss,
        "tangent drift alignment loss": drift_alignment_loss_test,
        "diffeomorphism loss1": diffeomorphism_loss1_test,
        "variance logdetg loss": variance_logdet_loss,
        "orthogonal coordinate loss": orthogonal_coordinates_error
    }


if __name__ == "__main__":
    # usage examples
    d = select_device("cuda")  # tries to use cuda
    print(d)

    d = select_device("mps")  # tries to use mps
    print(d)

    d = select_device("tpu")  # invalid device, falls back to available options
    print(d)

    d = select_device("cpu")  # invalid device, falls back to available options
    print(d)

    d = select_device()  # no preference, uses available devices in the predefined order
    print(d)
