if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sympy as sp
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from shillml.utils import fit_model2, process_data, set_grad_tracking, compute_test_losses
    from shillml.losses import DiffusionLoss3, DriftMSELoss3, LatentDriftMSE
    from shillml.losses.loss_modules import TotalLoss, LossWeights
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.models.autoencoders import AutoEncoder1
    from shillml.models.nsdes import AutoEncoderDrift, AutoEncoderDiffusion2, LatentNeuralSDE
    from shillml.pointclouds.dynamics import SDECoefficients

    # Inputs (same as before, adjust as needed)
    train_seed = 17
    norm = "fro"
    test_seed = None
    torch.manual_seed(train_seed)
    num_points = 30
    batch_size = num_points // 2
    num_test = 100
    a = -2
    b = 2
    epsilon = 0.2
    input_dim, latent_dim = 3, 2
    hidden_layers = [16]
    diffusion_layers = [16]
    drift_layers = [16]
    lr = 0.01
    epochs_ae = 10000
    epochs_diffusion = 10000
    epochs_drift = 10000
    ntime = 10000
    npaths = 5
    tn = 1.0
    weights = LossWeights()
    weights.encoder_contraction_weight = 0.0
    weights.decoder_contraction_weight = 0.0
    weights.tangent_angle_weight = 0.01
    weights.tangent_drift_weight = 0.01
    weights.diffeomorphism_reg1 = 0.0
    weight_decay = 0.0
    diffusion_tangent_drift_weight = 0.01
    rank_scale = 1.0
    c1, c2 = 5, 5
    bounds = [(a, b), (a, b)]
    large_bounds = [(a - epsilon, b + epsilon), (a - epsilon, b + epsilon)]
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()
    drift_act = nn.CELU()
    diffusion_act = nn.CELU()

    # Store loss information
    results = []

    # Define the symbols for the surfaces
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])

    # Define surfaces and dynamics configurations
    surfaces = {
        "Product": u * v / c1,
        # "Rational": (u + v) / (1 + u**2 + v**2),
        "Paraboloid": (u / c1) ** 2 + (v / c2) ** 2
        # "MixtureGaussians": (0.5 * sp.exp(-((u + 0.9) ** 2 + (v + 0.9) ** 2) / 1.6) +
        #                      0.5 * sp.exp(-((u - 0.9) ** 2 + (v - 0.9) ** 2) / 1.6)),
    }

    dynamics_configs = ["BM", "RBM", "DoubleWell", "Arbitrary"]

    for surface_name, fuv in surfaces.items():
        chart = sp.Matrix([u, v, fuv])

        # Create manifold
        manifold = RiemannianManifold(local_coordinates, chart)
        coefs = SDECoefficients()

        for dynamics_name in dynamics_configs:
            print(f"Training on {surface_name} with {dynamics_name} dynamics")

            # Define dynamics based on the selected type
            if dynamics_name == "BM":
                local_drift = sp.Matrix([0, 0])
                local_diffusion = sp.Matrix([[1, 0], [0, 1]])
            elif dynamics_name == "RBM":
                local_drift = manifold.local_bm_drift()
                local_diffusion = manifold.local_bm_diffusion()
            elif dynamics_name == "DoubleWell":
                double_well_potential = sp.Matrix([4 * u * (u ** 2 - 1), 2 * v])
                local_drift = manifold.local_bm_drift() - 0.5 * manifold.metric_tensor().inv() * double_well_potential
                local_diffusion = manifold.local_bm_diffusion()
            elif dynamics_name == "Arbitrary":
                local_drift = sp.Matrix([-5 * (u - 0.5), 5 * (u - 1.0)])
                local_diffusion = sp.Matrix([[0.2 * v, 0.01 * u], [0, 0.1 * v]])

            # Generate the point cloud plus dynamics observations
            cloud = PointCloud(manifold, bounds, local_drift, local_diffusion, compute_orthogonal_proj=True)
            x, _, mu, cov, local_x = cloud.generate(num_points, seed=train_seed)
            x, mu, cov, p, orthogcomp, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)

            # Define and train AE model
            ae = AutoEncoder1(input_dim, latent_dim, hidden_layers, encoder_act, decoder_act)
            ae_loss = TotalLoss(weights, norm, rank_scale)

            # Compute pre-training losses
            pre_train_losses = compute_test_losses(ae, ae_loss, x, p, orthonormal_frame, cov, mu)

            # Train the model
            fit_model2(ae, ae_loss, x, targets=(p, orthonormal_frame, cov, mu), lr=lr, epochs=epochs_ae,
                       batch_size=batch_size, weight_decay=weight_decay)
            set_grad_tracking(ae, False)

            # Define diffusion model and train
            latent_sde = LatentNeuralSDE(latent_dim, hidden_layers, hidden_layers, drift_act, diffusion_act, None)
            model_diffusion = AutoEncoderDiffusion2(latent_sde, ae)
            dpi = ae.encoder.jacobian_network(x).detach()
            encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
            diffusion_loss = DiffusionLoss3(tangent_drift_weight=diffusion_tangent_drift_weight, norm=norm)
            fit_model2(model_diffusion, diffusion_loss, x, targets=(mu, cov, encoded_cov, orthonormal_frame),
                       lr=lr, epochs=epochs_diffusion, batch_size=batch_size)
            set_grad_tracking(latent_sde.diffusion_net, False)

            # Define drift model and train
            model_drift = AutoEncoderDrift(latent_sde, ae)
            drift_loss = LatentDriftMSE()
            fit_model2(model_drift, drift_loss, x, targets=(mu, encoded_cov), epochs=epochs_drift,
                       batch_size=batch_size)
            set_grad_tracking(latent_sde.drift_net, False)

            # Generate test data and compute post-training losses
            x_test, _, mu_test, cov_test, local_x_test = cloud.generate(num_points, seed=test_seed)
            x_test, mu_test, cov_test, p_test, orthogcomp_test, orthonormal_frame_test = process_data(
                x_test, mu_test, cov_test, d=2, return_frame=True)
            test_ae_losses = compute_test_losses(ae, ae_loss, x_test, p_test, orthonormal_frame_test, cov_test, mu_test)
            dpi_test = ae.encoder.jacobian_network(x_test).detach()
            encoded_cov_test = torch.bmm(torch.bmm(dpi_test, cov_test), dpi_test.mT)
            diffusion_loss_test = diffusion_loss.forward(
                ae_diffusion=model_diffusion, x=x_test,
                targets=(mu_test, cov_test, encoded_cov_test, orthonormal_frame_test)
            )
            drift_loss_test = drift_loss.forward(
                drift_model=model_drift, x=x_test, targets=(mu_test, encoded_cov_test)
            )

            # Store results
            results.append({
                "surface": surface_name,
                "dynamics": dynamics_name,
                "pre_train_losses": pre_train_losses,
                "test_ae_losses": test_ae_losses,
                "diffusion_loss_test": diffusion_loss_test.item(),
                "drift_loss_test": drift_loss_test.item()
            })

    # Generate LaTeX table
    with open("results_table.tex", "w") as f:
        f.write("\\begin{tabular}{l l r r r r r}\n")
        f.write("Surface & Dynamics & Pre-train AE Loss & Test AE Loss & Diffusion Loss & Drift Loss \\\\ \n")
        f.write("\\hline\n")
        for result in results:
            surface = result["surface"]
            dynamics = result["dynamics"]
            pre_loss = result["pre_train_losses"]['reconstruction loss']
            test_loss = result["test_ae_losses"]['reconstruction loss']
            diffusion_loss = result["diffusion_loss_test"]
            drift_loss = result["drift_loss_test"]
            f.write(f"{surface} & {dynamics} & {pre_loss:.4f} & {test_loss:.4f} & {diffusion_loss:.4f} & {drift_loss:.4f} \\\\ \n")
        f.write("\\end{tabular}\n")

    print("Results saved to results_table.tex")
