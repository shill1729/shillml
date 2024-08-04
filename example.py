import torch
import torch.nn as nn
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from shillml.utils import fit_model, process_data, set_grad_tracking
from shillml.losses import CUCHTBAELoss, DiffusionLoss, DriftMSELoss
from shillml.diffgeo import RiemannianManifold
from shillml.pointclouds import PointCloud
from shillml.models.autoencoders import CUCHTBAE
from shillml.models.nsdes import AutoEncoderDrift, AutoEncoderDiffusion, LatentNeuralSDE


def generate_data(bounds, c1, c2, num_points):
    u, v = sp.symbols("u v", real=True)
    local_coordinates = sp.Matrix([u, v])
    chart = sp.Matrix([u, v, (u / c1) ** 2 + (v / c2) ** 2])
    manifold = RiemannianManifold(local_coordinates, chart)
    local_drift = manifold.local_bm_drift() - manifold.metric_tensor().inv() * sp.Matrix([4 * u * (u ** 2 - 1), 2 * v])
    local_diffusion = manifold.local_bm_diffusion()
    cloud = PointCloud(manifold, bounds, local_drift, local_diffusion)
    x, _, mu, cov, _ = cloud.generate(num_points)
    x, mu, cov, p, orthogcomp = process_data(x, mu, cov, d=2)
    return x, mu, cov, p, orthogcomp, cloud


def define_ae_model(input_dim, latent_dim, hidden_layers, activation, regularization_weights):
    contractive_weight, hessian_weight, tangent_drift_weight, tangent_bundle_weight = regularization_weights
    ae = CUCHTBAE(input_dim, latent_dim, hidden_layers, activation(), activation())
    ae_loss = CUCHTBAELoss(contractive_weight=contractive_weight,
                           hessian_weight=hessian_weight,
                           tangent_drift_weight=tangent_drift_weight,
                           tangent_bundle_weight=tangent_bundle_weight)
    return ae, ae_loss


def define_diffusion_model(latent_dim, hidden_layers, activation):
    latent_sde = LatentNeuralSDE(latent_dim, hidden_layers, hidden_layers, activation(), activation(), activation())
    return latent_sde


def fit_diffusion_model(ae, latent_sde, x, mu, cov, epochs, batch_size, tangent_drift_weight):
    model_drift = AutoEncoderDrift(latent_sde, ae)
    model_diffusion = AutoEncoderDiffusion(latent_sde, ae)
    dpi = ae.encoder.jacobian_network(x).detach()
    encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
    diffusion_loss = DiffusionLoss(tangent_drift_weight=tangent_drift_weight)
    fit_model(model_diffusion, diffusion_loss, x, (mu, cov, encoded_cov), epochs=epochs, batch_size=batch_size)
    set_grad_tracking(latent_sde.diffusion_net, False)
    drift_loss = DriftMSELoss()
    fit_model(model_drift, drift_loss, x, mu, epochs=epochs, batch_size=batch_size)
    return model_drift, model_diffusion


def compute_test_loss(ae, model_drift, model_diffusion, x, mu, cov, p, orthogcomp):
    diffusion_loss = DiffusionLoss(tangent_drift_weight=1.)
    drift_loss = DriftMSELoss()
    ae_loss = CUCHTBAELoss()
    dpi = ae.encoder.jacobian_network(x).detach()
    encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)
    dl = diffusion_loss.forward(model_diffusion, x, (mu, cov, encoded_cov))
    drl = drift_loss(model_drift, x, mu)
    aeloss = ae_loss.forward(ae, x, (p, orthogcomp, mu, cov))
    return dl, drl, aeloss


def main():
    epsilon = 0.5
    bounds = [(-1, 1), (-1, 1)]
    large_bounds = [(-1 - epsilon, 1 + epsilon), (-1 - epsilon, 1 + epsilon)]
    c1, c2 = 10, 10
    num_points = 30
    num_test = 100
    input_dim, latent_dim = 3, 2
    hidden_layers = [32]
    sde_layers = [32]
    activation = nn.Tanh
    epochs = 10000
    batch_size = 20
    ntime = 8000
    npaths = 10
    tn = 1

    # Regularization weights: ctr, hess, drift, bundle
    regularization_weights = (0.01, 0.0, 0.0001, 0.0001)
    tangent_drift_weight = regularization_weights[2]

    x, mu, cov, p, orthogcomp, cloud = generate_data(bounds, c1, c2, num_points)
    ae, ae_loss = define_ae_model(input_dim, latent_dim, hidden_layers, activation, regularization_weights)
    fit_model(ae, ae_loss, x, targets=(p, orthogcomp, mu, cov), epochs=epochs, batch_size=batch_size)
    set_grad_tracking(ae, False)

    latent_sde = define_diffusion_model(latent_dim, sde_layers, activation)
    model_drift, model_diffusion = fit_diffusion_model(ae, latent_sde, x, mu, cov, epochs, batch_size, tangent_drift_weight)

    # Uncomment the following lines to visualize the results: test data
    x, mu, cov, p, orthogcomp, cloud = generate_data(large_bounds, c1, c2, num_test)
    # Test loss:
    diffusion_extrap_loss, drift_extrap_loss, ae_extrap_loss = compute_test_loss(ae, model_drift,
                                                                                 model_diffusion,
                                                                                 x, mu, cov, p,
                                                                                 orthogcomp)
    print("AE extrapolation loss = " + str(ae_extrap_loss.detach().numpy()))
    print("Drift extrapolation loss = " + str(drift_extrap_loss.detach().numpy()))
    print("Diffusion extrapolation loss = " + str(diffusion_extrap_loss.detach().numpy()))
    x_hat = ae.decoder(ae.encoder(x))
    mu_hat = model_drift(x_hat).detach()
    x_hat = x_hat.detach()
    x = x.detach()
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], mu[:, 0], mu[:, 1], mu[:, 2], normalize=True, length=0.1)
    ax.quiver(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], mu_hat[:, 0], mu_hat[:, 1], mu_hat[:, 2], normalize=True,
              length=0.1, color="red")
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="CUCHTBAE")
    plt.show()

    # Plot SDEs
    x0 = np.array([1., 0.])
    true_latent_paths = cloud.latent_sde.sample_ensemble(x0, tn, ntime, npaths)
    model_latent_paths = latent_sde.sample_paths(x0, tn, ntime, npaths)
    true_ambient_paths = np.zeros((npaths, ntime + 1, 3))
    model_ambient_paths = np.zeros((npaths, ntime + 1, 3))

    for j in range(npaths):
        model_ambient_paths[j, :, :] = ae.decoder(torch.tensor(model_latent_paths[j, :, :],
                                                               dtype=torch.float32)).detach().numpy()
        for i in range(ntime + 1):
            true_ambient_paths[j, i, :] = np.squeeze(cloud.np_phi(*true_latent_paths[j, i, :]))

    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")
    for i in range(npaths):
        ax.plot3D(true_ambient_paths[i, :, 0], true_ambient_paths[i, :, 1], true_ambient_paths[i, :, 2], c="black",
                  alpha=0.8)
        ax.plot3D(model_ambient_paths[i, :, 0], model_ambient_paths[i, :, 1], model_ambient_paths[i, :, 2], c="blue",
                  alpha=0.8)
    ae.plot_surface(-1, 1, grid_size=30, ax=ax, title="CUCHTBAE")
    plt.show()


if __name__ == "__main__":
    main()
