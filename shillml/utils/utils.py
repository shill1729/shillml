"""
    This module contains utility functions for PyTorch.

    Specifically, we have

    set_grad_tracking: turn on/off the parameters of a nn.Module
    select_device: choose the computational device: cpu or gpu (cuda, or mps)

"""
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Union, List, Tuple, Optional


def fit_model(model: nn.Module,
              loss: nn.Module,
              input_data: Tensor,
              targets: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None,
              lr: float = 0.001,
              epochs: int = 1000,
              print_freq: int = 1000,
              weight_decay: float = 0.,
              batch_size: int = None) -> None:
    """
    Trains the given model using the specified loss function and data.
    Assumes the input of the loss function is (model, input_data, extra1, extra2, ...) where extras can be a single
    tensor or a tuple of tensors.

    Args:
        model (nn.Module): The neural network model to be trained.
        loss (nn.Module): The loss function used for training.
        input_data (Tensor): Input data to the network model.
        targets (Union[Tensor, List[Tensor], Tuple[Tensor, ...]]): Labels/targets data for the network output.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of epochs to train the model. Defaults to 1000.
        print_freq (int, optional): Frequency of printing the training loss. Defaults to 1000.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 0.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to None (which means use the full dataset).
    Returns:
        None
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # If batch_size is None or larger than dataset, use the full dataset as one batch
    if batch_size is None or batch_size > len(input_data):
        batch_size = len(input_data)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(*[input_data, *targets]) if isinstance(targets, (list, tuple)) else TensorDataset(input_data,
                                                                                                           targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0.0  # Reset the epoch loss at the start of each epoch

        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0]
            extra_targets = batch[1:]  # Extract remaining tensors in the batch as extra targets
            extra_targets = extra_targets[0] if len(extra_targets) == 1 else extra_targets
            # This works for score-based. but will it work for AE?
            loss_value = loss(model, inputs, *[extra_targets])
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
