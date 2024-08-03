from typing import List, Callable, Optional, Any
from torch import Tensor
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

import time
import torch


class FeedForwardNeuralNet(nn.Module):
    """
    A feedforward neural network class that constructs a network with specified layers
    and applies given activation functions.

    Attributes:
        layers (nn.ModuleList): A list of linear layers in the network.
        activations (callable): The activation function applied to all but the final layer.
        input_dim (int): the input dimension
        output_dim (int): the output dimension

    Methods:
        forward(x):
            Passes input tensor `x` through the network, applying the specified activation functions.

        jacobian_network(x):
            Computes the Jacobian matrix of the network's output with respect to its input.

        hessian_network(x):
            Computes the Hessian matrix of the network's output with respect to its input, for each coordinate

        jacobian_network_for_paths(x):
            Computes the Jacobian matrix of the network's output with respect to its input. Here the input
            is assumed to be a batch of sample-paths.

        hessian_network_for_paths(x):
            Computes the Hessian matrix of the network's output with respect to its input, for each coordinate.
            Here the input is assumed to be a batch of sample-paths.

        tie_weights(x):
            Tie the weights of this network to another network via transpose (but not the biases).

    """

    def __init__(self, neurons: List[int], activations: List[Optional[Callable[..., Any]]]):
        """
        Initializes the FeedForwardNeuralNet with the given neurons and activation functions.

        Args:
            neurons (list): A list of integers where each integer represents the number of nodes in a layer.
            activations (callable): The list of activation functions to apply after each linear layer
        """
        super(FeedForwardNeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = activations
        self.input_dim = neurons[0]
        self.output_dim = neurons[-1]

        for i in range(len(neurons) - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
            x (Tensor): The input tensor to the network.

        Returns:
            Tensor: The output tensor after passing through the network and activation functions.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
        return x

    def jacobian_network(self, x: Tensor, method: str = "autograd"):
        """

        :param x:
        :param method:
        :return:
        """
        if method == "autograd":
            return self._jacobian_network_autograd(x)
        elif method == "exact":
            return self._jacobian_network_explicit(x)

    def _jacobian_network_autograd(self, x: Tensor) -> Tensor:
        """
        Computes the Jacobian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network. The tensor should have `requires_grad` enabled.

        Returns:
            Tensor: A tensor representing the Jacobian matrix of the network's output with respect to the input.
        """
        x = x.requires_grad_(True)
        y = self.forward(x)
        jacobian = []
        for i in range(self.output_dim):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1
            jacob = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True)[0]
            jacobian.append(jacob)
        jacobian = torch.stack(jacobian, dim=1)
        return jacobian

    def _jacobian_network_explicit(self, x: Tensor):
        """
        Computes the Jacobian matrix of the network's output with respect to its input explicitly.

        Args:
            x (Tensor): The input tensor to the network of shape (batch_size, input_dim).

        Returns:
            Tensor: A tensor representing the Jacobian matrix of the network's output
                          with respect to the input, of shape (batch_size, output_dim, input_dim).
        """
        # Ensure input is batched
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        # Get weights of each layer
        weights = [layer.weight for layer in self.layers]

        # Initialize Jacobian with the first layer's weight
        jacobian = weights[0].repeat(batch_size, 1, 1)

        # Forward pass to store intermediate values
        y = x
        z_values = []
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            z = layer(y)
            z_values.append(z.detach().requires_grad_())

            if activation is not None:
                y = activation(z_values[-1])
            else:
                y = z_values[-1]

        # Backward pass to compute Jacobian
        for i in range(1, len(self.layers)):
            z = z_values[i - 1]

            if self.activations[i - 1] is not None:
                # Use autograd to compute the derivative of the activation function
                a = self.activations[i - 1](z)
                diag = torch.autograd.grad(a.sum(), z, create_graph=False)[0]
                diag_term = torch.diag_embed(diag.view(batch_size, -1))
            else:
                diag_term = torch.eye(z.size(1)).unsqueeze(0).repeat(batch_size, 1, 1)

            next_term = torch.bmm(weights[i].repeat(batch_size, 1, 1), diag_term)
            jacobian = torch.bmm(next_term, jacobian)

        # Apply final activation if present
        if self.activations[-1] is not None:
            z = z_values[-1]
            a = self.activations[-1](z)
            diag = torch.autograd.grad(a.sum(), z, create_graph=False)[0]
            diag_term = torch.diag_embed(diag.view(batch_size, -1))
            jacobian = torch.bmm(diag_term, jacobian)

        return jacobian

    def hessian_network(self, x: Tensor):
        """
        Computes the Hessian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network of shape (n, d),
                              where n is the batch size and d is the input dimension.

        Returns:
            Tensor: A tensor representing the Hessian matrix of the network's output
                          with respect to the input, of shape (n, self.output_dim, d, d).
        """
        n, d = x.shape
        x.requires_grad_(True)
        y = self.forward(x)

        hessians = []
        for i in range(self.output_dim):
            # Compute first-order gradients
            first_grads = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]

            # Compute second-order gradients (Hessian)
            hessian_rows = []
            for j in range(d):
                hessian_row = torch.autograd.grad(first_grads[:, j].sum(), x, retain_graph=True)[0]
                hessian_rows.append(hessian_row)

            hessian = torch.stack(hessian_rows, dim=1)
            hessians.append(hessian)

        # Stack Hessians for each output dimension
        hessians = torch.stack(hessians, dim=1)

        return hessians

    def jacobian_network_for_paths(self, x: Tensor):
        """
        Computes the Jacobian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network of shape (num_paths, n+1, d). The tensor should have
            `requires_grad` enabled.

        Returns:
            Tensor: A tensor representing the Jacobian matrix of the network's output with respect to the input.
        """
        num_paths, n, d = x.size()
        x.requires_grad_(True)
        jacobian = torch.zeros(num_paths, n, d, self.output_dim)
        for i in range(num_paths):
            for j in range(n):
                x_ij = x[i, j, :].unsqueeze(0)  # shape (1, d)
                output = self.forward(x_ij)  # shape (1, D)
                for k in range(self.output_dim):
                    grad_outputs = torch.zeros_like(output)
                    grad_outputs[0, k] = 1
                    gradients = torch.autograd.grad(outputs=output, inputs=x_ij, grad_outputs=grad_outputs,
                                                    create_graph=True)[0]  # shape (1, d)
                    jacobian[i, j, :, k] = gradients.squeeze()
        return jacobian.transpose(2, 3)

    def hessian_network_for_paths(self, x: Tensor):
        """
        Computes the Hessian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network of shape (num_paths, n+1, d). The tensor should have
            `requires_grad` enabled.

        Returns:
            Tensor: A tensor representing the Hessian matrix of the network's output with respect to the input.
        """
        num_paths, n, d = x.shape
        x.requires_grad_(True)
        outputs = self.forward(x)

        hessians = torch.zeros(num_paths, n, self.output_dim, d, d)

        for i in range(num_paths):
            for j in range(n):
                for k in range(self.output_dim):
                    grad_outputs = torch.zeros_like(outputs)
                    grad_outputs[i, j, k] = 1
                    grads = \
                        torch.autograd.grad(outputs, x, grad_outputs=grad_outputs, create_graph=True,
                                            allow_unused=True)[0]
                    if grads is None:
                        continue
                    for r in range(d):
                        hessian_row = torch.autograd.grad(grads[i, j, r], x, retain_graph=True, allow_unused=True)[0]
                        if hessian_row is not None:
                            hessians[i, j, k, r, :] = hessian_row[i, j, :]

        return hessians

    def tie_weights(self, other: nn.Module):
        """
        Tie the weights of this network to the transpose of another one.
        """
        for layer_self, layer_other in zip(self.layers, reversed(other.layers)):
            if isinstance(layer_self, nn.Linear) and isinstance(layer_other, nn.Linear):
                with torch.no_grad():
                    layer_self.weight.copy_(layer_other.weight.t())


# Testing the class
def test_feed_forward_neural_net():
    # Simple test case: y = Ax + b
    a = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    b = torch.tensor([1.0, 2.0])
    net1 = FeedForwardNeuralNet([2, 2], [None])
    net1.layers[0].weight = nn.Parameter(a)
    net1.layers[0].bias = nn.Parameter(b)

    x = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output1 = net1(x)
    expected_output1 = torch.matmul(a, x.t()).t() + b
    assert torch.allclose(output1, expected_output1), "Simple linear test failed"

    jacobian1 = net1.jacobian_network(x)
    expected_jacobian1 = a.unsqueeze(0).expand(x.size(0), -1, -1)
    print(jacobian1)
    print(expected_jacobian1)
    assert torch.allclose(jacobian1, expected_jacobian1), "Jacobian test for linear network failed"

    # Test case: y = A2 * ReLU(A1 * x + b1) + b2
    a1 = torch.tensor([[1.0, -1.0], [2.0, 0.5]])
    b1 = torch.tensor([-1.0, 1.0])
    a2 = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    b2 = torch.tensor([0.5, -1.0])

    net2 = FeedForwardNeuralNet([2, 2, 2], [func.relu, None])
    net2.layers[0].weight = nn.Parameter(a1)
    net2.layers[0].bias = nn.Parameter(b1)
    net2.layers[1].weight = nn.Parameter(a2)
    net2.layers[1].bias = nn.Parameter(b2)

    x = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output2 = net2(x)
    expected_output2 = torch.matmul(a2, func.relu(torch.matmul(a1, x.t()).t() + b1).t()).t() + b2
    assert torch.allclose(output2, expected_output2), "Two-layer test with ReLU failed"

    jacobian2 = net2.jacobian_network(x)
    with torch.no_grad():
        relu_grad = (torch.matmul(a1, x.t()).t() + b1 > 0).float()
        expected_jacobian2 = a2 @ (relu_grad.unsqueeze(2) * a1.unsqueeze(0))
    assert torch.allclose(jacobian2, expected_jacobian2), "Jacobian test for two-layer network failed"

    print("The implementation passed tests for Ax+b and A2ReLU(A1 x + b1)+b1 on output values and jacobians")


def test_jacobian_shapes():
    # Test for single input
    net = FeedForwardNeuralNet([3, 4, 2], [func.tanh, None])
    x_single = torch.randn(1, 3, requires_grad=True)
    jacobian_single_auto = net.jacobian_network(x_single, method="autograd")
    jacobian_single_explicit = net.jacobian_network(x_single, method="exact")
    print(f"Autograd Jacobian shape for single input: {jacobian_single_auto.shape}")
    print(f"Explicit Jacobian shape for single input: {jacobian_single_explicit.shape}")

    # Test for batched input
    x_batched = torch.randn(5, 3, requires_grad=True)
    jacobian_batched_auto = net.jacobian_network(x_batched, method="autograd")
    jacobian_batched_explicit = net.jacobian_network(x_batched, method="exact")
    print(f"Autograd Jacobian shape for batched input: {jacobian_batched_auto.shape}")
    print(f"Explicit Jacobian shape for batched input: {jacobian_batched_explicit.shape}")

    # Check if results are close
    print("Single input results close:", torch.allclose(jacobian_single_auto, jacobian_single_explicit, atol=1e-5))
    print("Batched input results close:", torch.allclose(jacobian_batched_auto, jacobian_batched_explicit, atol=1e-5))

    assert torch.allclose(jacobian_single_auto, jacobian_single_explicit,
                          atol=1e-5), "faied Autograd Jacobian and Explicit Jabocian match"
    assert torch.allclose(jacobian_batched_auto, jacobian_batched_explicit,
                          atol=1e-5), "failed Batched Autograd Jacobian and Explicit Jabocian match"
    print(net.jacobian_network(x_single))
    print(net.jacobian_network_for_paths(x_single.unsqueeze(0)))


def test_weight_tying():
    # Define the structure of the two networks
    neurons1 = [15, 2, 9, 1, 10, 2]
    neurons2 = neurons1[::-1]

    # Initialize two networks with given structures
    net1 = FeedForwardNeuralNet(neurons1, [func.relu, func.relu, func.relu, func.tanh, None])
    net2 = FeedForwardNeuralNet(neurons2, [func.relu, func.relu, func.relu, func.tanh, None])

    # Initialize the weights of net1
    for layer in net1.layers:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)

    # Tie the weights of net2 to net1
    net2.tie_weights(net1)

    # Check if the weights of net2 are the transpose of the weights of net1
    for layer1, layer2 in zip(reversed(net1.layers), net2.layers):
        if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
            assert torch.allclose(layer1.weight.t(), layer2.weight), "Weight tying failed"

    print("Weight tying test passed")

    print("\nChecking forward passes now")
    print(net1(torch.ones(neurons1[0])))
    print(net2(torch.ones(neurons1[-1])))


def test_feed_forward_neural_net_ensemble():
    a = torch.tensor([[2.0, 3.0], [4.0, 5.0], [1, 2.]])
    b = torch.tensor([1.0, 2.0, 1.])
    net1 = FeedForwardNeuralNet([2, 3], [None])
    net1.layers[0].weight = nn.Parameter(a)
    net1.layers[0].bias = nn.Parameter(b)

    num_paths, n, d = 5, 10, 2
    x = torch.randn(num_paths, n, d, requires_grad=True)
    output1 = net1(x.view(-1, d)).view(num_paths, n, -1)
    expected_output1 = torch.matmul(x.view(-1, d), a.t()).view(num_paths, n, -1) + b

    assert torch.allclose(output1, expected_output1), "Ensemble linear test failed"

    jacobian1 = net1.jacobian_network_for_paths(x)

    expected_jacobian1 = a.unsqueeze(0).unsqueeze(0).expand(num_paths, n, -1, -1)
    print(jacobian1[0, 0, :])
    print(expected_jacobian1[0, 0, :])
    assert torch.allclose(jacobian1, expected_jacobian1), "Jacobian test for ensemble linear network failed"

    print("The implementation passed tests for Ax+b on ensemble input values and jacobians")
    print(net1.hessian_network_for_paths(x).size())


def test_jacobian_performance():
    # Set up the network
    net = FeedForwardNeuralNet([10, 20, 15, 5], [func.relu, func.tanh, None])

    # Define batch sizes to test
    batch_sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000, 30000]

    # Initialize lists to store timing results
    autograd_times = []
    explicit_times = []

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 10, requires_grad=True)

        # Time autograd Jacobian
        start_time = time.time()
        jacobian_auto = net.jacobian_network(x, method="autograd")
        autograd_time = time.time() - start_time
        autograd_times.append(autograd_time)

        # Time explicit Jacobian
        start_time = time.time()
        jacobian_explicit = net.jacobian_network(x, method="exact")
        explicit_time = time.time() - start_time
        explicit_times.append(explicit_time)

        # Check that computations are close
        assert torch.allclose(jacobian_auto, jacobian_explicit,
                              atol=1e-5), f"Jacobians not close for batch size {batch_size}"

        print(f"Batch size: {batch_size}")
        print(f"Autograd time: {autograd_time:.4f}s")
        print(f"Explicit time: {explicit_time:.4f}s")
        print("Jacobians are close")
        print("--------------------")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, autograd_times, 'b-o', label='Autograd Jacobian')
    plt.plot(batch_sizes, explicit_times, 'r-o', label='Explicit Jacobian')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Computation Time (s)')
    plt.title('Jacobian Computation Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and print speedup
    speedup = np.array(autograd_times) / np.array(explicit_times)
    for i, batch_size in enumerate(batch_sizes):
        print(f"Batch size {batch_size}: Speedup = {speedup[i]:.2f}x")


def test_hessian_network():
    # Create a simple network
    net = FeedForwardNeuralNet([2, 3, 2], [func.relu, func.tanh, None])

    # Create a batch of inputs
    x = torch.randn(5, 2, requires_grad=True)

    # Compute Hessian
    hessian = net.hessian_network(x)

    # Check shape
    assert hessian.shape == (5, 2, 2, 2), f"Expected shape (5, 2, 2, 2), got {hessian.shape}"

    # Check symmetry
    assert torch.allclose(hessian, hessian.transpose(2, 3)), "Hessian is not symmetric"

    print("Hessian test passed")


# Run the tests
if __name__ == "__main__":
    import sympy as sp

    test_feed_forward_neural_net()
    test_feed_forward_neural_net_ensemble()
    test_jacobian_shapes()
    test_weight_tying()
    test_jacobian_performance()
    test_hessian_network()


    def hessian_function(f, x: Tensor):
        """
        Computes the Hessian matrix of a given function with respect to its input.

        Args:
            f (callable): The function to compute the Hessian for.
            x (Tensor): The input tensor of shape (n, d),
                              where n is the batch size and d is the input dimension.

        Returns:
            Tensor: A tensor representing the Hessian matrix of the function's output
                          with respect to the input, of shape (n, output_dim, d, d).
        """
        n, d = x.shape
        x.requires_grad_(True)
        y = f(x)
        output_dim = y.shape[1] if len(y.shape) > 1 else 1

        hessians = []
        for i in range(output_dim):
            # Compute first-order gradients
            first_grads = torch.autograd.grad(y[:, i].sum() if output_dim > 1 else y.sum(), x, create_graph=True)[0]

            # Compute second-order gradients (Hessian)
            hessian_rows = []
            for j in range(d):
                hessian_row = torch.autograd.grad(first_grads[:, j].sum(), x, retain_graph=True)[0]
                hessian_rows.append(hessian_row)

            hessian = torch.stack(hessian_rows, dim=1)
            hessians.append(hessian)

        # Stack Hessians for each output dimension
        hessians = torch.stack(hessians, dim=1)

        return hessians

    # Test functions
    def test_scalar_to_scalar():
        # f: R -> R, f(x) = x^2
        x_sym = sp.Symbol('x')
        f_sym = sp.cos(x_sym ** 2)

        # Sympy Hessian
        hessian_sym = sp.hessian(f_sym, (x_sym,))

        # Convert to numpy function
        # f_np = sp.lambdify(x_sym, f_sym, 'numpy')
        hessian_np = sp.lambdify(x_sym, hessian_sym, 'numpy')

        # PyTorch function
        def f_torch(x):
            return torch.cos(x ** 2)

        # Test
        x = torch.tensor([[2.0]], requires_grad=True)
        hessian_torch = hessian_function(f_torch, x)

        hessian_exact = hessian_np(x.item())

        assert np.allclose(hessian_torch.detach().numpy(), hessian_exact, atol=1e-6)
        print("Scalar to scalar test passed.")


    def test_vector_to_scalar():
        # g: R^2 -> R, g(x, y) = x^2 + xy + y^2
        x, y = sp.symbols('x y')
        g_sym = x ** 2 + x * y + y ** 2

        # Sympy Hessian
        hessian_sym = sp.hessian(g_sym, (x, y))

        # Convert to numpy function
        g_np = sp.lambdify((x, y), g_sym, 'numpy')
        hessian_np = sp.lambdify((x, y), hessian_sym, 'numpy')

        # PyTorch function
        def g_torch(x):
            return x[:, 0] ** 2 + x[:, 0] * x[:, 1] + x[:, 1] ** 2

        # Test
        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        hessian_torch = hessian_function(g_torch, x)

        hessian_exact = hessian_np(x[0, 0].item(), x[0, 1].item())

        assert np.allclose(hessian_torch.detach().numpy(), hessian_exact, atol=1e-6)
        print("Vector to scalar test passed.")


    def test_vector_to_vector():
        # h: R^3 -> R^2, h(x, y, z) = (x^2 + y^2, y*z + x)
        x, y, z = sp.symbols('x y z')
        h_sym = sp.Matrix([x ** 2 + y ** 2, y * z + x])

        # Sympy Hessian
        hessian_sym = [sp.hessian(h_sym[i], (x, y, z)) for i in range(2)]

        # Convert to numpy function
        h_np = sp.lambdify((x, y, z), h_sym, 'numpy')
        hessian_np = [sp.lambdify((x, y, z), hessian_sym[i], 'numpy') for i in range(2)]

        # PyTorch function
        def h_torch(x):
            return torch.stack([x[:, 0] ** 2 + x[:, 1] ** 2, x[:, 1] * x[:, 2] + x[:, 0]], dim=1)

        # Test
        x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        hessian_torch = hessian_function(h_torch, x)

        hessian_exact = np.stack([hessian_np[i](x[0, 0].item(), x[0, 1].item(), x[0, 2].item()) for i in range(2)])

        assert np.allclose(hessian_torch.detach().numpy(), hessian_exact, atol=1e-6)
        print("Vector to vector test passed.")


    # Run tests
    test_scalar_to_scalar()
    test_vector_to_scalar()
    test_vector_to_vector()
