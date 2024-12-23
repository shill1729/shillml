import torch
import torch.nn as nn
import sympy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shillml.utils import fit_model, fit_model2, process_data, set_grad_tracking, compute_test_losses
from shillml.losses.loss_modules import TotalLoss, LossWeights
from shillml.diffgeo import RiemannianManifold
from shillml.pointclouds import PointCloud
from shillml.models.autoencoders import AutoEncoder1


def main():
    # Inputs
    train_seed = 17
    norm = "fro"
    test_seed = None
    torch.manual_seed(train_seed)
    num_points = 100
    batch_size = num_points
    num_test = 100
    # Boundary for point cloud
    a = 0
    b = 2 * np.pi
    epsilon = 0.1
    input_dim, latent_dim = 2, 1
    hidden_layers = [32]
    lr = 0.01
    epochs_ae = 20000
    # weights for different penalties
    weights = LossWeights()
    weights.encoder_contraction_weight = 0.
    weights.decoder_contraction_weight = 0.
    weights.tangent_angle_weight = 0.
    weights.tangent_drift_weight = 0.
    weights.diffeomorphism_reg1 = 0.02
    weight_decay = 0.
    # weight for diffusion training drift alignment penalty:
    rank_scale = 1.
    # Flattening factors
    c1, c2 = 5, 5
    bounds = [(a, b)]
    large_bounds = [(a - epsilon, b + epsilon)]
    # Activation functions
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()

    # Define the manifold
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u])
    # Product
    radius = 1 + 0.8 * sp.cos(2 * u)
    xu = sp.cos(u) * radius
    yu = sp.sin(u) * radius
    # Creating the manifold
    chart = sp.Matrix([xu, yu])
    manifold = RiemannianManifold(local_coordinates, chart)
    # BM
    local_drift = sp.Matrix([0])
    local_diffusion = sp.Matrix([[1]])
    # Generate the point cloud plus dynamics observations
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
    # returns points, weights, drifts, cov, local coord
    x, _, mu, cov, local_x = cloud.generate(num_points, seed=train_seed)
    x, mu, cov, p, orthogcomp, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)

    # Define and train AE model
    ae = AutoEncoder1(input_dim, latent_dim, hidden_layers, encoder_act, decoder_act)
    ae_loss = TotalLoss(weights, norm, rank_scale)

    # Print pre-training losses
    print("Pre-training losses")
    pre_train_losses = compute_test_losses(ae, ae_loss, x, p, orthonormal_frame, cov, mu)
    for key, value in pre_train_losses.items():
        print(f"{key} = {value:.4f}")

    # Train the model
    print("\nTraining Autoencoder")
    fit_model2(ae, ae_loss, x, targets=(p, orthonormal_frame, cov, mu), lr=lr, epochs=epochs_ae,
               batch_size=batch_size, weight_decay=weight_decay)
    set_grad_tracking(ae, False)

    # Now all three components have been trained, we assess the performance on the test set.
    # returns points, weights, drifts, cov, local coord
    x_test, _, mu_test, cov_test, local_x_test = cloud.generate(num_test, seed=test_seed)
    x_test, mu_test, cov_test, p_test, orthogcomp_test, orthonormal_frame_test = process_data(x_test,
                                                                                              mu_test,
                                                                                              cov_test,
                                                                                              d=2,
                                                                                              return_frame=True)
    # Print post-training losses on the testing set for the AE
    print("Testing-set losses for the Autoencoder:")
    test_ae_losses = compute_test_losses(ae, ae_loss, x_test, p_test, orthonormal_frame_test, cov_test, mu_test)
    for key, value in test_ae_losses.items():
        print(f"{key} = {value:.4f}")

    z_test = ae.encoder(x_test).detach()
    x_test = x_test.detach()

    u1 = np.pi / 2
    u2 = 3 * np.pi / 2
    x1 = cloud.np_phi(u1).reshape(1, 2)
    x2 = cloud.np_phi(u2).reshape(1, 2)
    z1 = ae.encoder(torch.tensor(x1, dtype=torch.float32)).detach()
    z2 = ae.encoder(torch.tensor(x2, dtype=torch.float32)).detach()
    fig = plt.figure()
    ax = plt.subplot(211)
    ax.scatter(x_test[:, 0], x_test[:, 1])
    ax.scatter(x1[0, 0], x1[0, 1], c="red")
    ax.scatter(x2[0, 0], x2[0, 1], c="red")
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="Reconstruction", dim=2)
    plt.grid()

    ax = plt.subplot(212)
    ax.scatter(z_test[:], np.zeros(num_test))
    ax.scatter(z1, 0, c="red")
    ax.scatter(z2, 0, c="red")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
