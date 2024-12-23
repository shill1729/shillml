"""
This module, example_ae_dyn.py fits a three-stage model

1. AE with dynamics-penalities
2. Diffusion coefficient/ covariance
3. Drift

"""
# TODO: 12/5/2024
# 1. Embed a surface randomly into higher dimenison like D=10 or 20 and perform the algorithm
# 2. Get long time statistics on langevins, short time on bms
# 3. More ablation testing and this normalization problem bug
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sympy as sp
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from shillml.utils import fit_model, fit_model2, process_data, set_grad_tracking, compute_test_losses
    from shillml.losses import DiffusionLoss3, DriftMSELoss3, LatentDriftMSE
    from shillml.losses.loss_modules import TotalLoss, LossWeights
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.models.autoencoders import AutoEncoder1
    from shillml.models.nsdes import AutoEncoderDrift, AutoEncoderDiffusion2, LatentNeuralSDE
    from shillml.pointclouds.dynamics import SDECoefficients

    # Inputs
    train_seed = 17
    norm = "fro"
    test_seed = 42
    torch.manual_seed(train_seed)
    num_points = 30
    batch_size = num_points
    num_test = 100
    # Boundary for point cloud
    a = -1
    b = 1
    epsilon = 0.05
    input_dim, latent_dim = 3, 2
    hidden_layers = [16]
    diffusion_layers = [16]
    drift_layers = [16]
    lr = 0.01
    weight_decay = 0.
    # EPOCHS FOR TRAINING
    epochs_ae = 10000
    epochs_diffusion = 10000
    epochs_drift = 20000
    # PATHS PARAMETERS
    ntime = 1000
    npaths = 5
    tn = 1.
    # weights for different penalties
    weights = LossWeights()
    weights.encoder_contraction_weight = 0.
    weights.decoder_contraction_weight = 0.
    weights.tangent_angle_weight = 0.
    weights.tangent_drift_weight = 0.
    weights.diffeomorphism_reg1 = 0.
    rank_scale = 1.
    # weight for diffusion training drift alignment penalty:
    latent_cov_weight = 1.
    ambient_cov_weight = 0.
    diffusion_tangent_drift_weight = 0.
    normalize = False
    # Generate a fixed random matrix R with a seed
    R_seed = 123  # Set your seed
    torch.manual_seed(R_seed)
    D = 10  # Desired embedding dimension
    R = torch.randn(D, 3)  # Random matrix R of size (D, 3)

    # Flattening factors
    c1, c2 = 3, 3
    bounds = [(a, b), (a, b)]
    large_bounds = [(a - epsilon, b + epsilon), (a - epsilon, b + epsilon)]
    # Activation functions
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()
    drift_act = nn.Tanh()
    diffusion_act = nn.Tanh()

    # Define the manifold
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])

    # Here is a bunch of surfaces: undo the comment for one you want
    # Product
    # fuv = u * v / c1

    # Rational function
    # fuv = (u+v)/(1+u**2+v**2)

    # Paraboloid
    fuv = (u/c1)**2+(v/c2)**2

    # # Mixture of Gaussians
    # sigma_2 = 0.8
    # fuv = (0.5 * sp.exp(-((u + 0.9) ** 2 + (v + 0.9) ** 2) / (2 * sigma_2)) / (np.sqrt(2 * np.pi * sigma_2)) +
    #        0.5 * sp.exp(-((u - 0.9) ** 2 + (v - 0.9) ** 2) / (2 * sigma_2)) / (np.sqrt(2 * np.pi * sigma_2)))

    # For the form (u,v, f(u,v))
    chart = sp.Matrix([u, v, fuv])

    # For general parametric forms
    # Sphere
    # chart = sp.Matrix([sp.sin(u) * sp.cos(v), sp.sin(u) * sp.sin(v), sp.cos(u)])

    # initialize the manifold
    manifold = RiemannianManifold(local_coordinates, chart)
    coefs = SDECoefficients()

    # BM
    # local_drift = sp.Matrix([0, 0])
    # local_diffusion = sp.Matrix([[1, 0], [0, 1]])

    # RBM
    # local_drift = manifold.local_bm_drift()
    # local_diffusion = manifold.local_bm_diffusion()

    # Langevin with double well potential
    # double_well_potential = sp.Matrix([4 * u * (u ** 2 - 1), 2 * v])/4
    # local_drift = manifold.local_bm_drift() - 0.5 * manifold.metric_tensor().inv() * double_well_potential
    # local_diffusion = manifold.local_bm_diffusion()

    # Some arbitrary motion
    # local_drift = sp.Matrix([-5 * (u - 0.5), 5 * (u - 1.)]) / 2
    # local_diffusion = sp.Matrix([[0.1 * sp.sin(u) + v ** 2, 0.05 * sp.cos(v) + u],
    #                              [0.02 * u * v, 0.1 + 0.1 * v]
    #                              ]) / 5

    # Another arbitrary diffusion
    local_drift = sp.Matrix([u*v, -sp.sin(u)])/3
    local_diffusion = sp.Matrix([[u-v, u*v], [u+v, sp.sin(u)*v]])/2

    # Generate the point cloud plus dynamics observations
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
    # returns points, weights, drifts, cov, local coord
    x, _, mu, cov, local_x = cloud.generate(num_points, seed=train_seed)
    x, mu, cov, p, _, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)

    # Define and train AE model
    ae = AutoEncoder1(input_dim, latent_dim, hidden_layers, encoder_act, decoder_act)
    ae_loss = TotalLoss(weights, norm, normalize=normalize, scale=rank_scale)

    # Print pre-training losses
    print("Pre-training losses")
    pre_train_losses = compute_test_losses(ae, ae_loss, x, p, orthonormal_frame, cov, mu)
    for key, value in pre_train_losses.items():
        print(f"{key} = {value:.4f}")

    # Train the model
    print("\nTraining Autoencoder")
    fit_model(model=ae,
              loss=ae_loss,
              input_data=x,
              targets=(p, orthonormal_frame, cov, mu),
              lr=lr,
              epochs=epochs_ae,
              batch_size=batch_size,
              weight_decay=weight_decay)

    # Schedule for learning rate
    # fit_model2(model=ae,
    #            loss=ae_loss,
    #            input_data=x,
    #            targets=(p, orthonormal_frame, cov, mu),
    #            lr=lr,
    #            epochs=epochs_ae,
    #            batch_size=batch_size,
    #            weight_decay=weight_decay)
    set_grad_tracking(ae, False)

    # Define diffusion model
    latent_sde = LatentNeuralSDE(latent_dim, hidden_layers, hidden_layers, drift_act, diffusion_act, None)

    # Fit diffusion model
    model_diffusion = AutoEncoderDiffusion2(latent_sde, ae)
    dpi = ae.encoder.jacobian_network(x).detach()
    encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
    diffusion_loss = DiffusionLoss3(latent_cov_weight=latent_cov_weight,
                                    ambient_cov_weight=ambient_cov_weight,
                                    tangent_drift_weight=diffusion_tangent_drift_weight,
                                    norm="fro",
                                    normalize=normalize)
    print("\nTraining diffusion")
    fit_model(model_diffusion,
              diffusion_loss,
              x,
              targets=(mu, cov, encoded_cov, orthonormal_frame),
              lr=lr, epochs=epochs_diffusion,
              batch_size=batch_size)
    set_grad_tracking(latent_sde.diffusion_net, False)

    # Define and train the drift model
    model_drift = AutoEncoderDrift(latent_sde, ae)
    # drift_loss = DriftMSELoss3()
    drift_loss = LatentDriftMSE()
    print("\nTraining drift")
    fit_model(model_drift,
              drift_loss,
              x,
              targets=(mu, encoded_cov),
              epochs=epochs_drift,
              batch_size=batch_size)
    set_grad_tracking(latent_sde.drift_net, False)

    # Now all three components have been trained, we assess the performance on the test set.
    # returns points, weights, drifts, cov, local coord
    # Generate the point cloud plus dynamics observations
    cloud = PointCloud(manifold, large_bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
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

    # Compute diffusion losses for testing set:
    dpi_test = ae.encoder.jacobian_network(x_test).detach()
    encoded_cov_test = torch.bmm(torch.bmm(dpi_test, cov_test), dpi_test.mT)
    diffusion_loss_test = diffusion_loss.forward(ae_diffusion=model_diffusion,
                                                 x=x_test,
                                                 targets=(mu_test, cov_test, encoded_cov_test, orthonormal_frame_test)
                                                 )
    drift_loss_test = drift_loss.forward(drift_model=model_drift,
                                         x=x_test,
                                         targets=(mu_test, encoded_cov_test))
    print("\nSDE losses")
    print("Diffusion extrapolation loss = " + str(diffusion_loss_test.detach().numpy()))
    print("Drift extrapolation loss = " + str(drift_loss_test.detach().numpy()))

    # Plot SDEs
    z0_true = x[0, :2].detach()
    x0 = x[0, :]
    z0 = ae.encoder.forward(x0).detach()
    true_latent_paths = cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
    model_latent_paths = latent_sde.sample_paths(z0, tn, ntime, npaths)
    true_ambient_paths = np.zeros((npaths, ntime + 1, 3))
    model_ambient_paths = np.zeros((npaths, ntime + 1, 3))

    for j in range(npaths):
        model_ambient_paths[j, :, :] = ae.decoder(
            torch.tensor(model_latent_paths[j, :, :], dtype=torch.float32)).detach().numpy()
        for i in range(ntime + 1):
            true_ambient_paths[j, i, :] = np.squeeze(cloud.np_phi(*true_latent_paths[j, i, :]))

    x_test = x_test.detach()
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    for i in range(npaths):
        ax.plot3D(true_ambient_paths[i, :, 0], true_ambient_paths[i, :, 1], true_ambient_paths[i, :, 2], c="black",
                  alpha=0.8)
        ax.plot3D(model_ambient_paths[i, :, 0], model_ambient_paths[i, :, 1], model_ambient_paths[i, :, 2], c="blue",
                  alpha=0.8)
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="Reconstruction")
    ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2])
    plt.show()

    # # Plot densities of first coordinate of terminal:
    # # Extract the first coordinate at the terminal time for all paths
    # for i in range(ae.extrinsic_dim):
    #     true_first_coord_terminal = true_ambient_paths[:, -1, i]
    #     model_first_coord_terminal = model_ambient_paths[:, -1, i]
    #
    #     # Plot the KDEs
    #     plt.figure(figsize=(10, 6))
    #     sns.kdeplot(true_first_coord_terminal, label='True', color='black', fill=True)
    #     sns.kdeplot(model_first_coord_terminal, label='Model', color='blue', fill=True)
    #
    #     # Customize the plot
    #     plt.title('KDEs of the ' + str(i + 1) + '-th Coordinate at Terminal Time')
    #     plt.xlabel('First Coordinate at Terminal Time')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
