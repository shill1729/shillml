# sdepy
<!-- badges: start -->
<!-- badges: end -->
This is a package for all my ML algorithms/models.


## Installation

You can install the package via from GitHub on windows/mac in command line with:

``` 
python -m pip install git+https://github.com/shill1729/shillml.git
```

## Features
The package has functions/classes for

### Feed Forward Neural Networks:
Features:
1. Feedforward Neural Network: Create customizable feedforward neural networks with specified layers and activation functions.
2. Device Management: Easily switch between CPU and GPU for computations.
3. Jacobian Computation: Compute the Jacobian matrix of the network's output with respect to its input using either autograd or explicit methods.
4. Hessian Computation: Compute the Hessian matrix for batched input data.
5. Weight Tying: Tie the weights of two networks for symmetrical architectures.
6. Support for Batched Inputs: Efficiently handle computations for batched input data.


```python
import torch
import torch.nn.functional as F
from shillml import FeedForwardNeuralNet, get_device
# Define a network with 2 input neurons, 3 hidden neurons, and 1 output neuron
neurons = [2, 3, 1]
activations = [F.tanh, F.tanh, None]
net = FeedForwardNeuralNet(neurons, activations)
# Set device
device = get_device("cpu")
net.to(device)
# Forward pass
x = torch.tensor([[1.0, 2.0]], requires_grad=True).to(device)
output = net(x)
print("Output:", output)
# Jacobian of network
jacobian = net.jacobian_network(x, method="autograd")
print("Jacobian:", jacobian)
```