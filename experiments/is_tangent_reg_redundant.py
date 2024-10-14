# TODO add hypothesis testing for one tailed two sample test whether
#  mu_recon - mu_tangent > 0 or not
if __name__ == "__main__":
    import torch.nn as nn
    import torch
    import sympy as sp
    import numpy as np
    import pandas as pd
    import scipy.stats as spstats
    from shillml.utils import fit_model2, process_data, compute_test_losses, set_grad_tracking
    from shillml.losses.loss_modules import TotalLoss
    from shillml.diffgeo import RiemannianManifold
    from shillml.pointclouds import PointCloud
    from shillml.models.autoencoders import AutoEncoder1

    # Significance level
    alpha = 0.05
    num_train_points = 50
    batch_size = int(num_train_points/2)
    num_test_points = 100
    hidden_layers = [64]
    num_trials = 60  # Number of trials for each surface
    lr = 0.01
    num_epochs = 20000
    a, b = -2, 2  # Bounds can now be adjusted dynamically
    c1, c2 = 5., 5.
    epsilon = 0.75
    # Network architecture, other params
    input_dim, latent_dim = 3, 2
    encoder_act = nn.Tanh()
    decoder_act = nn.Tanh()

    # TODO change this to give [u,v,f(u,v)] or [x(u,v), y(u,v), z(u,v)] instead
    # Surface configurations
    surfaces = [
        # Rational function
        lambda u, v, c1, c2: 0.1*(u + v) / (1 + u ** 2 + v ** 2),
        # Product
        lambda u, v, c1, c2: u * v / c1,
        # Paraboloid
        lambda u, v, c1, c2: (u / c1) ** 2 + (v / c2) ** 2,
        # Hyperbolic paraboloid
        lambda u, v, c1, c2: (u / c1) ** 2 - (v / c2) ** 2,
        # Mixture of Gaussians
        lambda u, v, c1, c2: (
                0.5 * sp.exp(-((u + 0.9) ** 2 + (v + 0.9) ** 2) / (2 * 0.8)) / (np.sqrt(2 * np.pi * 0.8)) +
                0.5 * sp.exp(-((u - 0.9) ** 2 + (v - 0.9) ** 2) / (2 * 0.8)) / (np.sqrt(2 * np.pi * 0.8))
        )
    ]

    num_surfaces = len(surfaces)  # Number of surfaces to test
    all_results = []
    # Now compute statistics (mean, variance, median, max) over all trials for each surface
    surface_stats = {}
    for surface_func in surfaces:
        trial_results = []
        for trial in range(num_trials):

            # Define the manifold
            u, v = sp.symbols("u v", real=True)
            local_coordinates = sp.Matrix([u, v])
            fuv = surface_func(u, v, c1, c2)  # Call surface function with u, v
            chart = sp.Matrix([u, v, fuv])
            manifold = RiemannianManifold(local_coordinates, chart)

            # BM coefficients
            local_drift = sp.Matrix([0, 0])
            local_diffusion = sp.Matrix([[1, 0], [0, 1]])

            # Dynamic point cloud bounds based on surface function input
            bounds = [(a, b), (a, b)]
            large_bounds = [(a - epsilon, b + epsilon), (a - epsilon, b + epsilon)]

            # Generate the point cloud and dynamics for training
            cloud = PointCloud(manifold, bounds, local_drift, local_diffusion)
            x, _, mu, cov, local_x = cloud.generate(num_train_points, seed=None)
            x, mu, cov, p, orthogcomp, orthonormal_frame = process_data(x, mu, cov, d=2, return_frame=True)

            # AE model
            ae = AutoEncoder1(input_dim, latent_dim, hidden_layers, encoder_act, decoder_act)
            weights = {
                "reconstruction": 1.,
                "rank_penalty": 0.,
                "contractive_reg": 0.,
                "decoder_contractive_reg": 0.,
                "tangent_bundle": 0.,
                "drift_alignment": 0.,
                "diffeo_reg1": 0.,
                "diffeo_reg2": 0.,
                "variance_logdet": 0.,
                "orthogonal": 0.,
                "tangent_angles": 0.,
                "normal_component_recon": 0.
            }
            ae_loss = TotalLoss(weights, "fro", 0.)

            # Training
            fit_model2(ae, ae_loss, x, targets=(p, orthonormal_frame, cov, mu), lr=lr, epochs=num_epochs,
                       batch_size=batch_size)

            # Generate new points for testing (test set)
            cloud_test = PointCloud(manifold, large_bounds, local_drift, local_diffusion)
            x_test, _, mu_test, cov_test, _ = cloud_test.generate(num_test_points, seed=None)
            x_test, mu_test, cov_test, p_test, orthogcomp_test, orthonormal_frame_test = process_data(x_test,
                                                                                                      mu_test,
                                                                                                      cov_test,
                                                                                                      d=2,
                                                                                                      return_frame=True)

            # Evaluate losses after training
            losses = compute_test_losses(ae, ae_loss, x_test, p_test, orthonormal_frame_test, cov_test, mu_test)
            trial_results.append(losses)
        all_results.append(trial_results)
        # Now compute statistics (mean, variance, median, max) over all trials for each surface
        for i, surface_trials in enumerate(all_results):
            stats = {
                key: {
                    "mean": np.mean([trial[key] for trial in surface_trials]),
                    "std": np.std([trial[key] for trial in surface_trials]),
                    "median": np.median([trial[key] for trial in surface_trials]),
                    "max": np.max([trial[key] for trial in surface_trials]),
                } for key in surface_trials[0].keys()
            }
            surface_stats[f"surface_{i + 1}"] = stats
        print(surface_stats)

    # Create a list to store all the DataFrames
    dfs = []
    for surface, stats in surface_stats.items():
        # Convert the stats dictionary into a DataFrame
        df = pd.DataFrame(stats).T  # Transpose to have rows as loss types and columns as statistics
        dfs.append(df)
        print(f"\nStatistics for {surface}:")
        print(df)

        # One-tailed two sample Hypothesis testing for mean_Recon > mean_Tangent
        # Extract the means, standard deviations, and sample size from your DataFrame
        mean_rec_loss = df.loc['reconstruction loss', 'mean']
        std_rec_loss = df.loc['reconstruction loss', 'std']
        mean_tan_loss = df.loc['tangent angle loss', 'mean']
        std_tan_loss = df.loc['tangent angle loss', 'std']
        n = num_trials

        # Calculate the t-statistic
        t_statistic = (mean_rec_loss - mean_tan_loss) / np.sqrt((std_rec_loss ** 2 / n) + (std_tan_loss ** 2 / n))

        # Calculate the degrees of freedom using Welch-Satterthwaite equation
        df_welch = ((std_rec_loss ** 2 / n) + (std_tan_loss ** 2 / n)) ** 2 / (
                    (std_rec_loss ** 2 / n) ** 2 / (n - 1) + (std_tan_loss ** 2 / n) ** 2 / (n - 1))

        # Compute the one-tailed p-value
        p_value = 1 - spstats.t.cdf(t_statistic, df=df_welch)
        # Critical value
        t_critical = spstats.t.ppf(1 - alpha, df_welch)

        print(f"T-statistic: {t_statistic}")
        print(f"Critical region: {t_critical}")
        print(f"Degrees of Freedom: {df_welch}")
        print(f"One-tailed P-value: {p_value}")
        # Decision rule
        if t_statistic > t_critical:
            print("Reject the null hypothesis (H0: recon loss <= tangent loss) at "+str(alpha)+"-significance")
        else:
            print("Fail to reject the null hypothesis (H0: recon loss <= tangent loss) at "+str(alpha)+"-significance")

        # Generate LaTeX code for each surface
        latex_table = df.to_latex(float_format="%.6f", index=True)  # Adjust the format as needed
        print(f"\nLaTeX table for {surface}:")
        print(latex_table)

    # Now calculate the average across all the DataFrames
    # Concatenate all the DataFrames along the axis=0 (rows), then group by index and compute the mean
    average_df = pd.concat(dfs).groupby(level=0).mean()

    print("\nAverage Statistics across all surfaces:")
    print(average_df)
    # Generate LaTeX code for the average table
    average_latex_table = average_df.to_latex(float_format="%.6f", index=True)  # Adjust the format as needed
    print("\nLaTeX table for the average statistics across all surfaces:")
    print(average_latex_table)


