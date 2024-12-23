import torch
import torch.nn as nn
import sympy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from shillml.utils import fit_model, process_data, set_grad_tracking, compute_test_losses
from shillml.losses import DiffusionLoss3, DriftMSELoss3, LatentDriftMSE
from shillml.losses.loss_modules import TotalLoss, LossWeights
from shillml.diffgeo import RiemannianManifold
from shillml.pointclouds import PointCloud
from shillml.models.autoencoders import AutoEncoder1
from shillml.models.nsdes import AutoEncoderDrift, AutoEncoderDiffusion2, LatentNeuralSDE
from shillml.pointclouds.dynamics import SDECoefficients


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    train_seed: int = 17
    test_seed: int = 42
    num_points: int = 30
    num_test: int = 100
    input_dim: int = 3
    latent_dim: int = 2
    hidden_layers: List[int] = None
    diffusion_layers: List[int] = None
    drift_layers: List[int] = None
    lr: float = 0.01
    weight_decay: float = 0.0
    epochs_ae: int = 10000
    epochs_diffusion: int = 10000
    epochs_drift: int = 20000
    ntime: int = 1000
    npaths: int = 5
    tn: float = 1.0
    bounds: Tuple[Tuple[float, float], ...] = None
    epsilon: float = 0.05

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [16]
        if self.diffusion_layers is None:
            self.diffusion_layers = [16]
        if self.drift_layers is None:
            self.drift_layers = [16]
        if self.bounds is None:
            self.bounds = [(-1, 1), (-1, 1)]


@dataclass
class RegularizationWeights:
    """Weights for different regularization terms"""
    encoder_contraction: float = 0.001
    decoder_contraction: float = 0.
    tangent_angle: float = 0.005
    tangent_drift: float = 0.01
    diffeomorphism_reg1: float = 0.01
    latent_cov: float = 1.0
    ambient_cov: float = 0.0
    diffusion_tangent_drift: float = 0.01


class ManifoldGenerator:
    """Generates manifold and SDE coefficients"""

    @staticmethod
    def create_paraboloid(c1: float = 3, c2: float = 3) -> Tuple[RiemannianManifold, sp.Matrix, sp.Matrix]:
        u, v = sp.symbols("u v", real=True)
        local_coordinates = sp.Matrix([u, v])
        fuv = (u / c1) ** 2 + (v / c2) ** 2
        chart = sp.Matrix([u, v, fuv])
        manifold = RiemannianManifold(local_coordinates, chart)

        # Define SDE coefficients
        local_drift = sp.Matrix([u * v, -sp.sin(u)]) / 3
        local_diffusion = sp.Matrix([[u - v, u * v], [u + v, sp.sin(u) * v]]) / 2

        return manifold, local_drift, local_diffusion


class SDEAutoencoder:
    """Main class for training and evaluation"""

    def __init__(self, config: ExperimentConfig, weights: RegularizationWeights):
        self.config = config
        self.weights = weights
        self.setup_components()

    def setup_components(self):
        """Initialize all necessary components"""
        torch.manual_seed(self.config.train_seed)

        # Create manifold and generate data
        self.manifold, self.local_drift, self.local_diffusion = ManifoldGenerator.create_paraboloid()
        self.cloud = PointCloud(
            self.manifold,
            self.config.bounds,
            self.local_drift,
            self.local_diffusion,
            compute_orthogonal_proj=True
        )

        # Setup models
        self.ae = AutoEncoder1(
            self.config.input_dim,
            self.config.latent_dim,
            self.config.hidden_layers,
            nn.Tanh(),
            nn.Tanh()
        )

        self.latent_sde = LatentNeuralSDE(
            self.config.latent_dim,
            self.config.hidden_layers,
            self.config.hidden_layers,
            nn.Tanh(),
            nn.Tanh(),
            None
        )

    def generate_data(self, num_points: int, seed: int) -> Tuple:
        """Generate training or test data"""
        x, _, mu, cov, local_x = self.cloud.generate(num_points, seed=seed)
        return process_data(x, mu, cov, d=2, return_frame=True)

    def train(self) -> Dict[str, float]:
        """Train all components and return metrics"""
        # Generate training data
        x, mu, cov, p, orthogcomp, orthonormal_frame = self.generate_data(
            self.config.num_points,
            self.config.train_seed
        )

        # Setup losses
        weights = LossWeights()
        weights.encoder_contraction_weight = self.weights.encoder_contraction
        weights.decoder_contraction_weight = self.weights.decoder_contraction
        weights.tangent_angle_weight = self.weights.tangent_angle
        weights.tangent_drift_weight = self.weights.tangent_drift
        weights.diffeomorphism_reg1 = self.weights.diffeomorphism_reg1

        ae_loss = TotalLoss(weights, "fro", normalize=False)

        # Train autoencoder
        fit_model(
            self.ae,
            ae_loss,
            x,
            (p, orthonormal_frame, cov, mu),
            self.config.lr,
            self.config.epochs_ae,
            self.config.num_points,
            self.config.weight_decay
        )
        set_grad_tracking(self.ae, False)

        # Train diffusion
        model_diffusion = AutoEncoderDiffusion2(self.latent_sde, self.ae)
        dpi = self.ae.encoder.jacobian_network(x).detach()
        encoded_cov = torch.bmm(torch.bmm(dpi, cov), dpi.mT)

        diffusion_loss = DiffusionLoss3(
            latent_cov_weight=self.weights.latent_cov,
            ambient_cov_weight=self.weights.ambient_cov,
            tangent_drift_weight=self.weights.diffusion_tangent_drift,
            norm="fro",
            normalize=False
        )

        fit_model(
            model_diffusion,
            diffusion_loss,
            x,
            (mu, cov, encoded_cov, orthonormal_frame),
            self.config.lr,
            self.config.epochs_diffusion,
            self.config.num_points
        )
        set_grad_tracking(self.latent_sde.diffusion_net, False)

        # Train drift
        model_drift = AutoEncoderDrift(self.latent_sde, self.ae)
        drift_loss = LatentDriftMSE()

        fit_model(
            model_drift,
            drift_loss,
            x,
            (mu, encoded_cov),
            lr=self.config.lr,
            epochs=self.config.epochs_drift,
            batch_size=self.config.num_points
        )
        set_grad_tracking(self.latent_sde.drift_net, False)

        # Evaluate on test data
        test_metrics = self.evaluate()
        return test_metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data"""
        x_test, mu_test, cov_test, p_test, _, orthonormal_frame_test = self.generate_data(
            self.config.num_test,
            self.config.test_seed
        )

        # Compute all losses
        ae_loss = TotalLoss(LossWeights(), "fro", normalize=False)
        test_ae_losses = compute_test_losses(
            self.ae,
            ae_loss,
            x_test,
            p_test,
            orthonormal_frame_test,
            cov_test,
            mu_test
        )

        # Compute SDE losses
        dpi_test = self.ae.encoder.jacobian_network(x_test).detach()
        encoded_cov_test = torch.bmm(torch.bmm(dpi_test, cov_test), dpi_test.mT)

        model_diffusion = AutoEncoderDiffusion2(self.latent_sde, self.ae)
        diffusion_loss = DiffusionLoss3(
            latent_cov_weight=self.weights.latent_cov,
            ambient_cov_weight=self.weights.ambient_cov,
            tangent_drift_weight=self.weights.diffusion_tangent_drift,
            norm="fro",
            normalize=False
        )

        diffusion_loss_test = diffusion_loss(
            model_diffusion,
            x_test,
            (mu_test, cov_test, encoded_cov_test, orthonormal_frame_test)
        )

        model_drift = AutoEncoderDrift(self.latent_sde, self.ae)
        drift_loss = LatentDriftMSE()
        drift_loss_test = drift_loss(
            model_drift,
            x_test,
            (mu_test, encoded_cov_test)
        )

        metrics = {
            **test_ae_losses,
            'diffusion_loss': diffusion_loss_test.item(),
            'drift_loss': drift_loss_test.item()
        }

        return metrics


def run_ablation_study():
    """Run ablation study with different regularization settings"""
    base_config = ExperimentConfig()

    # Define different regularization settings to test
    ablation_settings = [
        {'name': 'all_regularizations', 'weights': RegularizationWeights()},
        {'name': 'no_contractions', 'weights': RegularizationWeights(encoder_contraction=0.0, decoder_contraction=0.0)},
        {'name': 'no_tangents', 'weights': RegularizationWeights(tangent_angle=0.0, tangent_drift=0.0)},
        {'name': 'no_diffeo', 'weights': RegularizationWeights(diffeomorphism_reg1=0.0)},
        {'name': 'no_regularization', 'weights': RegularizationWeights(
            encoder_contraction=0.0,
            decoder_contraction=0.0,
            tangent_angle=0.0,
            tangent_drift=0.0,
            diffeomorphism_reg1=0.0
        )}
    ]

    results = []
    for setting in ablation_settings:
        print(f"\nRunning experiment: {setting['name']}")
        model = SDEAutoencoder(base_config, setting['weights'])
        metrics = model.train()
        results.append({
            'setting': setting['name'],
            **metrics
        })

    # Create results DataFrame
    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    results_df = run_ablation_study()

    # Display results
    print("\nAblation Study Results:")
    print(results_df.to_string(index=False))

    # # Create a heatmap visualization
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(
    #     results_df.set_index('setting').select_dtypes(include=[np.number]),
    #     annot=True,
    #     fmt='.3f',
    #     cmap='viridis'
    # )
    # plt.title('Ablation Study Results')
    # plt.tight_layout()
    # plt.show()