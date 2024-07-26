from shillml.autoencoders import AutoEncoder
import matplotlib.pyplot as plt
from PIL import Image
import io
import os


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
