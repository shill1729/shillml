"""
This script trains a DACTBAE model on a point cloud generated from a Riemannian manifold with a specified drift
and diffusion process. The trained autoencoder is then used to fit both a drift and a diffusion model in the latent
space using a latent neural SDE. The script computes and prints various extrapolation losses, including reconstruction,
drift, diffusion, and tangent drift errors, and visualizes the model's performance by plotting the predicted vs. true
paths in the ambient space. Additionally, it compares the SDE paths generated by the model against the true paths.
"""

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sympy as sp
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from shillml.utils import fit_model, process_data, set_grad_tracking
    from shillml.losses import NewAELoss, DiffusionLoss2, DriftMSELoss2
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.models.autoencoders import NewAE
    from shillml.models.nsdes import AutoEncoderDrift, AutoEncoderDiffusion2, LatentNeuralSDE
    from shillml.models.nsdes import ambient_quadratic_variation_drift
    from shillml.pointclouds.dynamics import SDECoefficients

    # Inputs
    train_seed = 17
    test_seed = None
    torch.manual_seed(train_seed)
    epsilon = 0.5
    bounds = [(-1, 1), (-1, 1)]
    large_bounds = [(-1 - epsilon, 1 + epsilon), (-1 - epsilon, 1 + epsilon)]
    c1, c2 = 5, 9
    num_points = 30
    num_test = 100
    input_dim, latent_dim = 3, 2
    hidden_layers = [64]
    sde_layers = [64]
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()
    drift_act = nn.GELU()
    diffusion_act = nn.GELU()
    lr = 0.0001
    epochs_ae = 10000
    epochs_diffusion = 100
    epochs_drift = 100
    batch_size = num_points
    ntime = 8000
    npaths = 30
    tn = 0.5
    contractive_weight = 0.00001
    tangent_drift_weight = 0.001
    tangent_bundle_weight = 0.001
    diffeo_weight = 0.00001

    # Define the manifold
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])
    chart = sp.Matrix(
        [u, v, -(u/10)**2+(v/10)**2])
    manifold = RiemannianManifold(local_coordinates, chart)
    coefs = SDECoefficients()
    # BM
    # local_drift = sp.Matrix([0, 0])
    # local_diffusion = sp.Matrix([[1, 0], [0, 1]])
    # RBM
    # local_drift = manifold.local_bm_drift()
    # local_diffusion = manifold.local_bm_diffusion()
    # # Langevin with double well potential
    local_drift = manifold.local_bm_drift() - 0.2*manifold.metric_tensor().inv() * sp.Matrix([4 * u * (u ** 2 - 1), 2 * v])
    local_diffusion = manifold.local_bm_diffusion()*coefs.diffusion_circular()

    # Generate the point cloud plus dynamics observations
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion)
    x, _, mu, cov, _ = cloud.generate(num_points, seed=train_seed)  # returns points, weights, drifts, cov, local coord
    x, mu, cov, p, orthogcomp, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)

    # Define AE model
    ae = NewAE(input_dim, latent_dim, hidden_layers, encoder_act, decoder_act)
    ae_loss = NewAELoss(contractive_weight=contractive_weight,
                        tangent_drift_weight=tangent_drift_weight,
                        tangent_bundle_weight=tangent_bundle_weight,
                        diffeo_weight=diffeo_weight
                        )
    fit_model(ae, ae_loss, x, targets=(p, orthogcomp, mu, cov, orthonormal_frame), lr=lr, epochs=epochs_ae, batch_size=batch_size)
    set_grad_tracking(ae, False)

    # Define diffusion model
    latent_sde = LatentNeuralSDE(latent_dim, hidden_layers, hidden_layers, drift_act, diffusion_act, None)

    # Fit diffusion model
    model_drift = AutoEncoderDrift(latent_sde, ae)
    model_diffusion = AutoEncoderDiffusion2(latent_sde, ae)
    dpi = ae.encoder.jacobian_network(x).detach()
    encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
    diffusion_loss = DiffusionLoss2(tangent_drift_weight=tangent_drift_weight)
    fit_model(model_diffusion,
              diffusion_loss,
              x,
              targets=(mu, cov, encoded_cov, orthonormal_frame),
              lr=lr, epochs=epochs_diffusion,
              batch_size=batch_size)
    set_grad_tracking(latent_sde.diffusion_net, False)
    drift_loss = DriftMSELoss2()
    fit_model(model_drift,
              drift_loss,
              x,
              targets=(mu, encoded_cov),
              epochs=epochs_drift,
              batch_size=batch_size)

    # Compute test loss
    x, _, mu, cov, _, = cloud.generate(num_test, seed=test_seed)
    x, mu, cov, p, orthogcomp, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)
    dpi = ae.encoder.jacobian_network(x).detach()
    encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
    dl = diffusion_loss.forward(model_diffusion, x, (mu, cov, encoded_cov, orthonormal_frame))
    drl = drift_loss(model_drift, x, (mu, encoded_cov))
    aeloss = ae_loss.forward(ae, x, (p, orthogcomp, mu, cov, orthonormal_frame))

    # Calculate error
    z = ae.encoder(x)
    latent_diff = model_diffusion.latent_sde.diffusion(z)
    bbt = torch.bmm(latent_diff, latent_diff.mT)
    decoder_hess = ae.decoder_hessian(z)
    q_model = ambient_quadratic_variation_drift(bbt, decoder_hess)
    q_true = torch.tensor(cloud.observed_q, dtype=torch.float32).squeeze()
    q_term_error = torch.mean(torch.linalg.vector_norm(q_true - q_model, ord=2, dim=1) ** 2)
    # Tangent drift error
    tangent_drift_true = torch.tensor(cloud.observed_tangent_drift, dtype=torch.float).squeeze()
    tangent_drift_model = model_drift.forward(x) - 0.5 * q_model
    tangent_drift_error = torch.mean(
        torch.linalg.vector_norm(tangent_drift_model - tangent_drift_true, ord=2, dim=1) ** 2)
    # error_bound
    error_bound = q_term_error + tangent_drift_error

    # Print results
    print("AE extrapolation loss = " + str(aeloss.detach().numpy()))
    print("Drift extrapolation loss = " + str(drl.detach().numpy()))
    print("Diffusion extrapolation loss = " + str(dl.detach().numpy()))
    print("q terms error = " + str(q_term_error.detach().numpy()))
    print("Tangent drift error " + str(tangent_drift_error.detach().numpy()))
    print("Sum = " + str(error_bound.detach().numpy()))

    # Visualization
    x_hat = ae.decoder(ae.encoder(x)).detach()
    mu_hat = model_drift(x_hat).detach()
    x = x.detach()
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], mu[:, 0], mu[:, 1], mu[:, 2], normalize=True, length=0.1)
    ax.quiver(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], mu_hat[:, 0], mu_hat[:, 1], mu_hat[:, 2], normalize=True,
              length=0.1, color="red")
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="New Model")
    plt.show()

    # Plot SDEs
    x0 = ae.encoder(x[0, :]).detach()
    true_latent_paths = cloud.latent_sde.sample_ensemble(x0, tn, ntime, npaths)
    model_latent_paths = latent_sde.sample_paths(x0, tn, ntime, npaths)
    true_ambient_paths = np.zeros((npaths, ntime + 1, 3))
    model_ambient_paths = np.zeros((npaths, ntime + 1, 3))

    for j in range(npaths):
        model_ambient_paths[j, :, :] = ae.decoder(
            torch.tensor(model_latent_paths[j, :, :], dtype=torch.float32)).detach().numpy()
        for i in range(ntime + 1):
            true_ambient_paths[j, i, :] = np.squeeze(cloud.np_phi(*true_latent_paths[j, i, :]))

    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    for i in range(npaths):
        ax.plot3D(true_ambient_paths[i, :, 0], true_ambient_paths[i, :, 1], true_ambient_paths[i, :, 2], c="black",
                  alpha=0.8)
        ax.plot3D(model_ambient_paths[i, :, 0], model_ambient_paths[i, :, 1], model_ambient_paths[i, :, 2], c="blue",
                  alpha=0.8)
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="New Model")
    plt.show()

    # Plot densities of first coordinate of terminal:
    # Extract the first coordinate at the terminal time for all paths
    for i in range(ae.extrinsic_dim):
        true_first_coord_terminal = true_ambient_paths[:, -1, i]
        model_first_coord_terminal = model_ambient_paths[:, -1, i]

        # Plot the KDEs
        plt.figure(figsize=(10, 6))
        sns.kdeplot(true_first_coord_terminal, label='True', color='black', fill=True)
        sns.kdeplot(model_first_coord_terminal, label='Model', color='blue', fill=True)

        # Customize the plot
        plt.title('KDEs of the '+str(i)+'-th Coordinate at Terminal Time')
        plt.xlabel('First Coordinate at Terminal Time')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
