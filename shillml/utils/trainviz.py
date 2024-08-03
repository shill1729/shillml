import io
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from shillml.autoencoders import AutoEncoder


class Callback:
    def on_epoch_end(self, epoch, model, metrics):
        pass

    def on_train_end(self, model):
        pass


class PlotSurfaceCallback(Callback):
    def __init__(self, num_frames, a, b, grid_size, save_dir):
        self.num_frames = num_frames
        self.a = a
        self.b = b
        self.grid_size = grid_size
        self.frames = []
        self.epoch_interval = None
        self.save_dir = save_dir

    def on_train_begin(self, total_epochs):
        self.epoch_interval = max(1, total_epochs // self.num_frames)

    def on_epoch_end(self, epoch, model: AutoEncoder, metrics):
        if (epoch + 1) % self.epoch_interval == 0 or epoch == 0:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            model.plot_surface(self.a, self.b, self.grid_size, ax=ax, title=f"Epoch {epoch + 1}")

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.frames.append(Image.open(buf))
            plt.close(fig)

    def on_train_end(self, model):
        if self.frames:
            gif_path = os.path.join(self.save_dir, 'training_animation.gif')
            self.frames[0].save(gif_path, save_all=True, append_images=self.frames[1:], duration=500, loop=0)
            print(f"Animation saved to: {gif_path}")


class VectorFieldCallback(Callback):
    def __init__(self, num_frames, x, mu, cov, P, save_dir='./vector_field_plots'):
        self.num_frames = num_frames
        self.x = x
        self.mu = mu
        self.cov = cov
        self.P = P
        self.save_dir = save_dir
        self.epoch_interval = None
        self.frames = []

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_begin(self, total_epochs):
        self.epoch_interval = max(1, total_epochs // self.num_frames)

    def on_epoch_end(self, epoch, model, metrics):
        if (epoch + 1) % self.epoch_interval == 0 or epoch == 0:
            self.plot_vector_field(epoch, model)

    def plot_vector_field(self, epoch, model):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the point cloud
        ax.scatter(self.x[:, 0].detach(), self.x[:, 1].detach(), self.x[:, 2].detach(), c='b', s=20, alpha=0.6)

        # Compute the vector field
        with torch.no_grad():
            x_hat, dpi, model_projection, decoder_hessian = model(self.x)
            bbt_proxy = torch.bmm(torch.bmm(dpi, self.cov), dpi.mT)
            qv = torch.stack(
                [torch.einsum("nii -> n", torch.bmm(bbt_proxy, decoder_hessian[:, i, :, :])) for i in
                 range(model.extrinsic_dim)])
            qv = qv.T
            tangent_vector = self.mu - 0.5 * qv
            normal_proj = torch.eye(model.extrinsic_dim).expand(self.x.size(0), model.extrinsic_dim,
                                                                model.extrinsic_dim) - self.P
            normal_proj_vector = torch.bmm(normal_proj, tangent_vector.unsqueeze(2)).squeeze(2)

        # Plot the vector field
        ax.quiver(self.x[:, 0].detach(), self.x[:, 1].detach(), self.x[:, 2].detach(),
                  normal_proj_vector[:, 0].detach(), normal_proj_vector[:, 1].detach(), normal_proj_vector[:, 2].detach(),
                  length=0.1, normalize=True, color='r', alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Vector Field (Epoch {epoch + 1})')

        # Save the plot
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.frames.append(Image.open(buf))
        plt.close(fig)

    def on_train_end(self, model):
        if self.frames:
            gif_path = os.path.join(self.save_dir, 'vector_field_animation.gif')
            self.frames[0].save(gif_path, save_all=True, append_images=self.frames[1:], duration=500, loop=0)
            print(f"Vector field animation saved to: {gif_path}")
