import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import List, Callable, Optional, Any, Union, Tuple
from torch import Tensor
from dataclasses import dataclass
from enum import Enum


class JacobianMethod(str, Enum):
    AUTOGRAD = "autograd"
    EXACT = "exact"


@dataclass
class NetworkOutput:
    output: Tensor
    jacobian: Optional[Tensor] = None
    hessian: Optional[Tensor] = None


class FeedForwardNeuralNet(nn.Module):
    """
    A feedforward neural network with advanced differential computation capabilities.

    Features:
    - Automatic or explicit Jacobian computation
    - Hessian computation
    - Support for batched inputs and path processing
    - Weight tying capabilities
    """

    def __init__(
            self,
            neurons: List[int],
            activations: List[Optional[Callable[..., Any]]],
            device: Optional[torch.device] = None
    ):
        """
        Initialize the network.

        Args:
            neurons: List of integers specifying layer dimensions
            activations: List of activation functions for each layer
            device: Target device for computation (CPU/GPU)
        """
        super().__init__()

        if len(neurons) < 2:
            raise ValueError("Network must have at least input and output layers")
        if len(neurons) - 1 != len(activations):
            raise ValueError("Number of activations must match number of layers minus 1")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList([
            nn.Linear(neurons[i], neurons[i + 1]).to(self.device)
            for i in range(len(neurons) - 1)
        ])
        self.activations = activations
        self.input_dim = neurons[0]
        self.output_dim = neurons[-1]

    def forward(
            self,
            x: Tensor,
            compute_jacobian: bool = False,
            compute_hessian: bool = False,
            jacobian_method: JacobianMethod = JacobianMethod.EXACT
    ) -> Union[Tensor, NetworkOutput]:
        """
        Forward pass with optional differential computations.

        Args:
            x: Input tensor
            compute_jacobian: Whether to compute Jacobian
            compute_hessian: Whether to compute Hessian
            jacobian_method: Method for Jacobian computation

        Returns:
            Either the output tensor or NetworkOutput containing derivatives
        """
        if not compute_jacobian and not compute_hessian:
            return self._forward_pass(x)

        output = self._forward_pass(x)
        jacobian = None
        hessian = None

        if compute_jacobian:
            jacobian = (
                self._jacobian_network_explicit(x)
                if jacobian_method == JacobianMethod.EXACT
                else self._jacobian_network_autograd(x)
            )

        if compute_hessian:
            hessian = self._compute_hessian(x)

        return NetworkOutput(output=output, jacobian=jacobian, hessian=hessian)

    def _forward_pass(self, x: Tensor) -> Tensor:
        """Internal forward pass implementation."""
        x = x.to(self.device)
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation is not None:
                x = activation(x)
        return x

    @torch.no_grad()
    def _compute_activation_gradient(self, z: Tensor, activation: Callable) -> Tensor:
        """Compute gradient of activation function."""
        z.requires_grad_(True)
        a = activation(z)
        grad = torch.autograd.grad(a.sum(), z, create_graph=False)[0]
        z.requires_grad_(False)
        return grad

    def _compute_hessian(self, x: Tensor) -> Tensor:
        """Compute Hessian matrix efficiently."""
        batch_size, input_dim = x.shape
        x.requires_grad_(True)
        y = self.forward(x)

        hessians = []
        for i in range(self.output_dim):
            first_grads = torch.autograd.grad(
                y[:, i].sum(), x, create_graph=True, retain_graph=True
            )[0]

            hessian_rows = torch.stack([
                torch.autograd.grad(
                    first_grads[:, j].sum(), x, retain_graph=True
                )[0]
                for j in range(input_dim)
            ], dim=1)

            hessians.append(hessian_rows)

        return torch.stack(hessians, dim=1)

    def tie_weights(self, other: 'FeedForwardNeuralNet') -> None:
        """
        Tie weights with another network for autoencoder architectures.
        The decoder weights are set as the transpose of the encoder weights.
        Both networks will update together during training.

        Args:
            other: Network to tie weights with (typically the encoder network)
        """
        for self_layer, other_layer in zip(self.layers, reversed(other.layers)):
            if isinstance(self_layer, nn.Linear) and isinstance(other_layer, nn.Linear):
                with torch.no_grad():
                    self_layer.weight.copy_(other_layer.weight.t())