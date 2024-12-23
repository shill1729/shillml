import sympy as sp
from shillml.utils import fit_model
from shillml.diffgeo import RiemannianManifold
from shillml.pointclouds import PointCloud
from shillml.utils import process_data
from shillml.losses.loss_modules import TotalLoss, LossWeights
from shillml.utils import compute_test_losses
from shillml.models.autoencoders import AutoEncoder1
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
seed = 17
torch.manual_seed(seed)
num_pts = 30
num_test = 1000
batch_size = 30
epochs = 1000
hidden_dims = [32, 32]
lr = 0.001
epsilon = 0.2
npaths = 2
tn = 10
ntime = 9000
a = -1
b = 1
weights = LossWeights()
weights.reconstruction = 1.
weights.tangent_drift_weight = 0.
weights.diffeomorphism_reg1 = 0.
weights.encoder_contraction_weight = 0.
bounds = [(a, b), (a, b)]
large_bounds = [(a - epsilon, b + epsilon), (a-epsilon, b + epsilon)]
# Generate data
u, v = sp.symbols("u v", real=True)
local_coordinates = sp.Matrix([u, v])
fuv = (u/2)**2+(v/2)**2
chart = sp.Matrix([u, v, fuv])

print("Computing geometry...")
manifold = RiemannianManifold(local_coordinates, chart)
# The local dynamics
print("Computing drift...")
local_drift = sp.Matrix([0,0])
print("Computing diffusion...")
local_diffusion = sp.eye(2,2)

# Generating the point cloud
cloud = PointCloud(manifold, bounds, local_drift, local_diffusion)
x, _, mu, cov, _ = cloud.generate(num_pts, seed=seed)
x, mu, cov, p, orthogcomp, frame = process_data(x, mu, cov, d=2, return_frame=True)
# Define model
ae = AutoEncoder1(3, 2, hidden_dims, nn.Tanh(), nn.Tanh())
ae_loss = TotalLoss(weights)
# Print results
pre_train_losses = compute_test_losses(ae, ae_loss, x, p, frame, cov, mu)
print("\n Pre-training losses")
for key, value in pre_train_losses.items():
    print(f"{key} = {value:.4f}")

# Fit the model
fit_model(ae, ae_loss, x, (p, frame, cov, mu), lr=lr, epochs=epochs, batch_size=batch_size)
# Test data:
cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion)
x_test, _, mu_test, cov_test, _ = cloud.generate(num_test, seed=None)
x_test, mu_test, cov_test, p_test, orthogcomp_test, frame_test = process_data(x_test, mu_test, cov_test, d=2,
                                                                              return_frame=True)
# Print results
test_losses = compute_test_losses(ae, ae_loss, x_test, p_test, frame_test, cov_test, mu_test)
print("\n Test losses")
for key, value in test_losses.items():
    print(f"{key} = {value:.4f}")

# Detach and plot
x = x.detach()
x_test = x_test.detach()

# Create a single figure with two subplots side by side
fig = plt.figure(figsize=(20, 10))

# First subplot
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(x[:, 0], x[:, 1], x[:, 2])
ae.plot_surface(-1, 1, grid_size=30, ax=ax1, title="ae - Training point cloud")

# Second subplot
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2])
ae.plot_surface(-1, 1, grid_size=30, ax=ax2, title="ae - Test point cloud")

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()