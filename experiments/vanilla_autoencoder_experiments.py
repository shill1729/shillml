"""
    Is the Jacobian of the decoder of a vanilla Autoencoder trained under reconstruction loss rank deficient?
"""

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    from shillml.utils import fit_model, fit_model2, process_data, set_grad_tracking, compute_test_losses
    from shillml.losses.loss_modules import TotalLoss, LossWeights
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.models.autoencoders import AutoEncoder1
    from shillml.pointclouds.dynamics import SDECoefficients
    from shillml.models.autoencoders.lae import computeW, pairwise_distances
    # weights
    # Inputs
    train_seed = 17
    norm = "fro"
    test_seed = None
    if train_seed is not None:
        torch.manual_seed(train_seed)
    num_points = 30
    batch_size = num_points
    num_test = 100
    # Boundary for point cloud
    a = -3.5
    b = 3.5
    epsilon = 1.
    bounds = [(a, b), (a, b)]
    large_bounds = [(a - epsilon, b + epsilon), (a - epsilon, b + epsilon)]
    # Network architecture:
    input_dim, latent_dim = 3, 2
    hidden_layers = [32]
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()
    # Learning rate schedule:
    lr = 0.01  # Initial learning rate
    scheduler_step_size = 2000  # The interval at which we reduce the learning rate
    gamma = 0.1  # The learning rate multiplicative decay factor
    # Training epochs
    epochs_ae = 1000
    rank_scale = 1.5  # Right now this is the target min-SV
    weight_decay = 0.
    weights = LossWeights()
    # Flattening factors for manifold
    c1, c2 = 2, 2

    # Define the manifold
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])
    # A rational function
    # fuv = (u+v)/(1+u**2+v**2)

    # Product
    # fuv = u*v/c1

    # Paraboloid
    fuv = (u/c1)**2+(v/c2)**2

    # Hyperbolic paraboloid:
    # fuv = (u / c1) ** 2 - (v / c2) ** 2

    # # Mixture of Gaussians
    # sigma_2 = 0.8
    # fuv = (0.5 * sp.exp(-((u + 0.9) ** 2 + (v + 0.9) ** 2) / (2 * sigma_2)) / (np.sqrt(2 * np.pi * sigma_2)) +
    #        0.5 * sp.exp(-((u - 0.9) ** 2 + (v - 0.9) ** 2) / (2 * sigma_2)) / (np.sqrt(2 * np.pi * sigma_2)))

    # Creating the manifold from a graph of a function
    chart = sp.Matrix([u, v, fuv])

    # Sphere
    # chart = sp.Matrix([sp.sin(u)*sp.cos(v), sp.sin(u)*sp.sin(v), sp.cos(u)])
    #
    # # Torus:
    # R, r = 2, 1
    # chart = sp.Matrix([(R + r * sp.cos(v)) * sp.cos(u),
    #                    (R + r * sp.cos(v)) * sp.sin(u),
    #                    r * sp.sin(v)])
    #
    # # Klein Bottle
    # chart = sp.Matrix([(R + sp.cos(u / 2) * sp.sin(v) - sp.sin(u / 2) * sp.sin(2 * v)) * sp.cos(u),
    #                    (R + sp.cos(u / 2) * sp.sin(v) - sp.sin(u / 2) * sp.sin(2 * v)) * sp.sin(u),
    #                    sp.sin(u / 2) * sp.sin(v) + sp.cos(u / 2) * sp.sin(2 * v)])
    #
    # # Mobius strip
    # R, w = 2, 1  # R: radius, w: width
    # chart = sp.Matrix([(R + v * sp.cos(u / 2)) * sp.cos(u),
    #                    (R + v * sp.cos(u / 2)) * sp.sin(u),
    #                    v * sp.sin(u / 2)])
    #
    # # Helicoid
    # a = 2  # pitch of the helicoid
    # chart = sp.Matrix([u * sp.cos(v), u * sp.sin(v), a * v])
    #
    # # Catenoid:
    # c = 2  # scaling factor
    # chart = sp.Matrix([c * sp.cosh(v / c) * sp.cos(u),
    #                    c * sp.cosh(v / c) * sp.sin(u),
    #                    v])
    #
    # # Ennepers surface:
    # chart = sp.Matrix([u - u ** 3 / 3 + u * v ** 2,
    #                    v - v ** 3 / 3 + v * u ** 2,
    #                    u ** 2 - v ** 2])

    # Monkey saddle
    # chart = sp.Matrix([u, v, u ** 3 - 3 * u * v ** 2])

    manifold = RiemannianManifold(local_coordinates, chart)
    coefs = SDECoefficients()

    # BM
    # local_drift = sp.Matrix([0, 0])
    # local_diffusion = sp.Matrix([[1, 0], [0, 1]])

    # RBM
    local_drift = manifold.local_bm_drift()
    local_diffusion = manifold.local_bm_diffusion()

    # # Langevin with double well potential
    # local_drift = manifold.local_bm_drift() - 0.2 * manifold.metric_tensor().inv() * sp.Matrix(
    #     [4 * u * (u ** 2 - 1), 2 * v])
    # local_diffusion = manifold.local_bm_diffusion() * coefs.diffusion_circular()*2

    # Generate the point cloud plus dynamics observations
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=False)
    # returns points, weights, drifts, cov, local coord
    x, _, mu, cov, local_x = cloud.generate(num_points, seed=train_seed)
    # Sphere adhoc
    # x = torch.randn(size=(num_points, 3))
    # x = x/torch.linalg.vector_norm(x, dim=1, keepdim=True)
    x, mu, cov, p, orthogcomp, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)

    # p_true = cloud.get_true_orthogonal_proj(local_x)
    # p_true = torch.tensor(p_true, dtype=torch.float32)
    # print("Error of SVD-P versus true P")
    # print(torch.mean(torch.linalg.matrix_norm(p-p_true, ord="fro")))

    # Compute graph laplacian / affinity matrix
    dist = pairwise_distances(x)
    affinity = computeW(dist, 0.01)

    # Define AE model
    ae = AutoEncoder1(input_dim, latent_dim, hidden_layers, encoder_act, decoder_act)
    ae_loss = TotalLoss(weights, norm, rank_scale, affinity=affinity)
    print("Pre-training losses")
    # Print results
    pre_train_losses = compute_test_losses(ae, ae_loss, x, p, orthonormal_frame, cov, mu)
    for key, value in pre_train_losses.items():
        print(f"{key} = {value:.4f}")
    print("\nTraining Autoencoder")
    # fit_model(ae, ae_loss, x, targets=(p, orthonormal_frame, cov, mu), lr=lr, epochs=epochs_ae,
    #           batch_size=batch_size, weight_decay=weight_decay)
    fit_model2(model=ae,
               loss=ae_loss,
               input_data=x,
               targets=(p, orthonormal_frame, cov, mu),
               lr=lr,
               epochs=epochs_ae,
               print_freq=1000,
               weight_decay=weight_decay,
               batch_size=batch_size,
               scheduler_step_size=scheduler_step_size,
               gamma=gamma)
    losses = compute_test_losses(ae, ae_loss, x, p, orthonormal_frame, cov, mu)
    print("\nAutoencoder losses/penalities")
    for key, value in losses.items():
        print(f"{key} = {value:.4f}")
    set_grad_tracking(ae, False)
    # Compute test loss
    cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion)
    x_test, _, mu_test, cov, _, = cloud.generate(num_test, seed=test_seed)
    # Sphere adhoc
    # x_test = torch.randn(size=(num_test, 3))
    # x_test = x_test / torch.linalg.vector_norm(x_test, dim=1, keepdim=True)
    x_test, mu_test, cov_test, p_test, orthogcomp_test, orthonormal_frame_test = process_data(x_test, mu_test, cov, d=2, return_frame=True)
    aeloss = ae_loss.forward(ae, x_test, (p_test, orthonormal_frame_test, cov_test, mu_test))
    # Print results
    losses = compute_test_losses(ae, ae_loss, x_test, p_test, orthonormal_frame_test, cov_test, mu_test)
    print("\nAutoencoder losses/penalities")
    for key, value in losses.items():
        print(f"{key} = {value:.4f}")

    print("AE extrapolation total weighted loss = " + str(aeloss.detach().numpy()))


    # Visualization
    z = ae.encoder(x)
    z_test = ae.encoder(x_test)
    min_train_rank = torch.min(torch.linalg.matrix_rank(ae.decoder_jacobian(z))).detach()
    min_test_rank = torch.min(torch.linalg.matrix_rank(ae.decoder_jacobian(z_test))).detach()
    print("Minimum training rank ="+str(min_train_rank))
    print("Minimum testing rank =" + str(min_train_rank))
    min_sivs = torch.linalg.matrix_norm(ae.decoder_jacobian(z), ord=-2).detach()
    min_svs_test = torch.linalg.matrix_norm(ae.decoder_jacobian(z_test), ord=-2).detach()
    print("Smallest min-SV over test set = "+str(torch.min(min_svs_test)))
    x_hat = ae.decoder(z).detach()
    x_hat_test = ae.decoder(z_test).detach()
    x = x.detach()
    x_test = x_test.detach()

    fig = plt.figure(figsize=(14, 6))

    # First plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2])
    scatter1 = ax1.scatter(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], c=min_sivs, cmap='viridis')
    fig.colorbar(scatter1, ax=ax1, label='Minimum Singular Values')
    ae.plot_surface(-1, 1, grid_size=30, ax=ax1, title="New Model - Training Data")

    # Second plot
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2])
    scatter2 = ax2.scatter(x_hat_test[:, 0], x_hat_test[:, 1], x_hat_test[:, 2], c=min_svs_test, cmap='viridis')
    fig.colorbar(scatter2, ax=ax2, label='Minimum Singular Values')
    ae.plot_surface(-1, 1, grid_size=30, ax=ax2, title="New Model - Test Data")

    plt.tight_layout()
    plt.show()

