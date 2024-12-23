import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_sde_animation(x, tn, ntime, ou, save_animation=False, seed=None):
    """Create an animation of the forward and reverse SDE flows with trails and timing"""
    # Calculate flows
    w = generate_brownian_motion(tn, ntime, d=3, seed=seed)
    forward_paths = flow(x, tn, w, ou.drift, ou.diffusion, ntime)
    x_flow = forward_paths[-1]
    reverse_paths = flow(x_flow, tn, w[::-1, :], ou.drift, ou.diffusion, ntime, True)

    # Combine paths for full animation
    all_paths = np.concatenate([forward_paths, reverse_paths])
    num_points = x.shape[0]
    dt = tn / ntime

    # Set up the figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#1C1C1C')
    ax.set_facecolor('#1C1C1C')

    # Set axis limits with some padding
    max_vals = np.max(np.abs(all_paths))
    ax.set_xlim(-max_vals * 1.1, max_vals * 1.1)
    ax.set_ylim(-max_vals * 1.1, max_vals * 1.1)
    ax.set_zlim(-max_vals * 1.1, max_vals * 1.1)
    ax.grid(True, alpha=0.2)

    # Initialize visualization elements
    scatter = ax.scatter([], [], [], c='cyan', s=30, alpha=0.8)
    trail_length = 50
    lines = [ax.plot([], [], [], 'cyan', alpha=0.2)[0] for _ in range(num_points)]

    # Add title, time display, and flow direction
    title = ax.set_title('3D Stochastic Flow', fontsize=12, color='white', pad=10)
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes,
                          color='white', fontsize=10, ha='left', va='top')
    direction_text = ax.text2D(0.02, 0.94, '', transform=ax.transAxes,
                               color='white', fontsize=10, ha='left', va='top')

    def init():
        """Initialize animation"""
        scatter._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return [scatter] + lines + [time_text, direction_text]

    def update(frame):
        """Update animation frame"""
        # Determine if we're in forward or reverse flow
        is_forward = frame < len(forward_paths)
        current_color = 'cyan' if is_forward else '#ff6b6b'  # cyan for forward, coral for reverse

        # Update scatter points
        scatter._offsets3d = (all_paths[frame, :, 0],
                              all_paths[frame, :, 1],
                              all_paths[frame, :, 2])
        scatter.set_color(current_color)

        # Update trails
        start_idx = max(0, frame - trail_length)
        for i in range(num_points):
            x_data = all_paths[start_idx:frame + 1, i, 0]
            y_data = all_paths[start_idx:frame + 1, i, 1]
            z_data = all_paths[start_idx:frame + 1, i, 2]
            lines[i].set_data(x_data, y_data)
            lines[i].set_3d_properties(z_data)
            lines[i].set_color(current_color)

            # Fade the trails
            alpha = np.linspace(0.05, 0.3, frame - start_idx + 1)[-1]
            lines[i].set_alpha(alpha)

        # Update time display and flow direction
        current_time = (frame % len(forward_paths)) * dt
        time_text.set_text(f'Time: {current_time:.3f}s')
        direction_text.set_text('Flow: Forward' if is_forward else 'Flow: Reverse')

        return [scatter] + lines + [time_text, direction_text]

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(all_paths),
                        init_func=init, interval=20,
                        blit=False)

    if save_animation:
        ani.save('sde_flow.gif', writer='pillow')

    plt.show()

    return ani

def generate_brownian_motion(t, num_steps, d=3, seed=None):
    dt = t / num_steps
    rng = np.random.default_rng(seed=seed)
    z = rng.normal(size=(num_steps, d)) * np.sqrt(dt)
    W = np.cumsum(z, axis=0)
    W = np.vstack((np.zeros(d), W))
    return W


def flow(points, t, W, mu, sigma, num_steps=1000, reverse=False):
    dt = t / num_steps
    paths = np.zeros((num_steps + 1, len(points), 3))
    paths[0] = points
    for i in range(num_steps):
        dW = W[i + 1] - W[i]
        for j in range(len(points)):
            x = paths[i, j]
            if reverse:
                mu1 = -mu(x)
                sigma1 = sigma(x)
            else:
                mu1 = mu(x)
                sigma1 = sigma(x)
            paths[i + 1, j] = paths[i, j] + mu1 * dt + (sigma1 @ dW)
    return paths


class OrnsteinUhlenbeck:
    def __init__(self, mean_reversion_speed=1., diffusion_coefficient=0.2, d=3):
        self.mean_reversion_speed = mean_reversion_speed
        self.diffusion_coefficient = diffusion_coefficient
        self.dim = d

    def drift(self, x):
        return -self.mean_reversion_speed * x

    def diffusion(self, x):
        return self.diffusion_coefficient * np.eye(self.dim)