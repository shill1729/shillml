"""
    This module contains utility functions for PyTorch.

    Specifically, we have

    set_grad_tracking: turn on/off the parameters of a nn.Module
    select_device: choose the computational device: cpu or gpu (cuda, or mps)

"""
import torch.nn as nn
import torch


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
    device = select_device("cuda")  # tries to use cuda
    print(device)

    device = select_device("mps")  # tries to use mps
    print(device)

    device = select_device("tpu")  # invalid device, falls back to available options
    print(device)

    device = select_device("cpu")  # invalid device, falls back to available options
    print(device)

    device = select_device()  # no preference, uses available devices in the predefined order
    print(device)
