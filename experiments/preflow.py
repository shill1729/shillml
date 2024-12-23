"""
This module implements sequential training of autoencoders:
1. Train vanilla AE on original data
2. Apply SDE flow to data
3. Train flow AE on flowed data
4. Compare reconstructions and reverse flows
"""
import numpy as np
import torch.nn as nn
import sympy as sp
import matplotlib.pyplot as plt
from shillml.utils import fit_model, process_data, compute_test_losses
from shillml.losses.loss_modules import TotalLoss, LossWeights
from shillml.diffgeo import RiemannianManifold
from shillml.pointclouds import PointCloud
from shillml.models.autoencoders import AutoEncoder1
from shillml.flows import create_sde_animation, flow, generate_brownian_motion, OrnsteinUhlenbeck
from shillml.pointclouds.dynamics import SDECoefficients
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parameters
    k = 4  # flow harmonic potential coefficient
    sigma = 0.02  # flow diffusion coefficient
    num_points = 50
    batch_size = num_points
    a, b = -1, 1
    input_dim, latent_dim = 3, 2
    hidden_layers = [32]
    lr = 0.001
    weight_decay = 0.
    epochs_ae = 5000
    ntime = 2000
    flow_time = 0.55
    train_seed = None

    # Setup manifold (example with torus, but could be any surface)
    u, v = sp.symbols("u v", real=True)
    # chart = sp.Matrix([(4 + 2 * sp.cos(u)) * sp.cos(v),
    #                    (4 + 2 * sp.cos(u)) * sp.sin(v),
    #                    2 * sp.sin(u)])
    chart = sp.Matrix([u, v, u**2+v**2])
    manifold = RiemannianManifold(sp.Matrix([u, v]), chart)
    local_drift = sp.Matrix([0, 0])
    local_diffusion = sp.Matrix([[1, 0], [0, 1]])
    # Generate point cloud
    cloud = PointCloud(manifold, [(a, b), (a, b)],
                       local_drift, local_diffusion,
                       compute_orthogonal_proj=True)
    x, _, mu, cov, local_x = cloud.generate(num_points, seed=train_seed)

    # 1. Train vanilla autoencoder on original data
    print("\nTraining Vanilla Autoencoder")
    x_processed, mu, cov, p, orthogcomp, orthonormal_frame = process_data(
        x, mu, cov, d=2, return_frame=True)

    vanilla_ae = AutoEncoder1(input_dim, latent_dim, hidden_layers,
                              nn.Tanh(), nn.Tanh())
    weights = LossWeights()
    weights.encoder_contraction_weight = 0.
    vanilla_loss = TotalLoss(weights, "fro")

    fit_model(model=vanilla_ae,
              loss=vanilla_loss,
              input_data=x_processed,
              targets=(p, orthonormal_frame, cov, mu),
              lr=lr,
              epochs=epochs_ae,
              batch_size=batch_size,
              weight_decay=weight_decay)

    vanilla_output = vanilla_ae.decoder(
        vanilla_ae.encoder(x_processed)).detach().numpy()

    # 2. Apply SDE flow
    ou = OrnsteinUhlenbeck(mean_reversion_speed=k, diffusion_coefficient=sigma)
    create_sde_animation(x, flow_time, 100, ou, False, None)
    flow_noise = generate_brownian_motion(flow_time, ntime, d=3, seed=train_seed)
    x_flow = flow(x, flow_time, flow_noise, ou.drift, ou.diffusion, ntime)[-1]
    x_reflow = flow(x_flow, flow_time, flow_noise[::-1],
                    ou.drift, ou.diffusion, ntime, True)[-1]

    # 3. Train flow autoencoder
    print("\nTraining Flow Autoencoder")
    x_flow_processed, mu, cov, p, orthogcomp, orthonormal_frame = process_data(
        x_flow, mu, cov, d=2, return_frame=True)

    flow_ae = AutoEncoder1(input_dim, latent_dim, hidden_layers,
                           nn.Tanh(), nn.Tanh())
    flow_loss = TotalLoss(weights, "fro")

    fit_model(model=flow_ae,
              loss=flow_loss,
              input_data=x_flow_processed,
              targets=(p, orthonormal_frame, cov, mu),
              lr=lr,
              epochs=epochs_ae,
              batch_size=batch_size,
              weight_decay=weight_decay)

    # Get flow AE output and apply reverse flow
    flow_ae_output = flow_ae.decoder(
        flow_ae.encoder(x_flow_processed)).detach().numpy()
    x_postflow = flow(flow_ae_output, flow_time, flow_noise[::-1],
                      ou.drift, ou.diffusion, ntime, True)[-1]

    # Plot results
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Original data and vanilla AE
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title("Vanilla Autoencoder")
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2], label="Original")
    ax1.scatter(vanilla_output[:, 0], vanilla_output[:, 1],
                vanilla_output[:, 2], c="red", label="Vanilla AE")
    vanilla_ae.plot_surface(-1, 1, grid_size=30, ax=ax1)
    ax1.legend()

    # Plot 2: Flow data and flow AE
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_title("Flow Autoencoder")
    ax2.scatter(x_flow[:, 0], x_flow[:, 1], x_flow[:, 2], label="Flow")
    ax2.scatter(flow_ae_output[:, 0], flow_ae_output[:, 1],
                flow_ae_output[:, 2], c="red", label="Flow AE")
    flow_ae.plot_surface(-1, 1, grid_size=30, ax=ax2)
    ax2.legend()

    # Plot 3: All flows comparison
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_title("Flow Comparison")
    ax3.scatter(x[:, 0], x[:, 1], x[:, 2], label="Original")
    ax3.scatter(x_flow[:, 0], x_flow[:, 1], x_flow[:, 2],
                c="red", label="Forward Flow")
    ax3.scatter(x_reflow[:, 0], x_reflow[:, 1], x_reflow[:, 2],
                c="green", label="True Reverse")
    ax3.scatter(x_postflow[:, 0], x_postflow[:, 1], x_postflow[:, 2],
                c="blue", label="Model Reverse")
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # Print reconstruction errors
    vanilla_error = np.mean(np.linalg.norm(x - vanilla_output, axis=1) ** 2)
    flow_error = np.mean(np.linalg.norm(x - x_postflow, axis=1) ** 2)
    print(f"\nVanilla AE reconstruction error: {vanilla_error:.4f}")
    print(f"Flow AE reconstruction error: {flow_error:.4f}")
