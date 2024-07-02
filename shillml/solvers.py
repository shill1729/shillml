import numpy as np


def euler_maruyama(x0, tn, mu, sigma, n=1000, bd=None, seed=None):
    """
    Solve an optionally stopped multivariate autonomous SDE using the Euler-Maruyama scheme.

    :param x0: vector shape (d,), initial point
    :param tn: float, time-horizon
    :param mu: function(x) -> vector shape (d,), drift vector
    :param sigma: function(x) -> matrix shape (d, m), diffusion matrix
    :param n: integer, number of time steps
    :param bd: list of 2 floats, a boundary box [a,b]^d, for stopping
    :param seed: integer, seed for RNG

    :return: vector, sample-path, possibly stopped at the boundary of a box
    """
    h = tn / n
    rng = np.random.default_rng(seed)
    m = sigma(x0).shape[1]
    z = rng.normal(size=(n, m))
    d = x0.shape[0]
    x = np.zeros((n+1, d))
    x[0, :] = x0
    for i in range(n):
        dB_h = z[i] * np.sqrt(h)
        diffusion_term = sigma(x[i, :]) @ dB_h
        x[i+1, :] = x[i, :] + mu(x[i, :]) * h + diffusion_term.reshape(d)
        # Stop at boundary
        if bd is not None:
            vb1 = np.linalg.norm(x[i + 1, :], ord=np.inf)
            vb2 = np.linalg.norm(x[i + 1, :], ord=-np.inf)
            if vb1 > bd[1] or vb2 < bd[0]:
                x[i:, :] = x[i, :]
                print("Process stopped at boundary")
                return x
    return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def sphere_orthog_proj(x):
        n = x/np.linalg.norm(x, ord=2)
        n = n.reshape(3, 1)
        orthog_comp = n @ n.T
        return np.identity(3)-orthog_comp

    tn = 5
    n = 50000
    seed = 17
    # Simulate a Brownian motion in space
    x0 = np.array([0, 0, 0])
    bm3 = euler_maruyama(x0, tn, lambda x: np.zeros(3), lambda x:np.identity(3), n, seed=seed)
    # Simulate a Langevin motion in space
    x0 = np.ones(3) * 3
    langevin3 = euler_maruyama(x0, tn, lambda x: -x * 10, lambda x: np.identity(3), n, seed=seed)
    # Simulate Brownian motion on the sphere
    x0 = np.array([1., 0., 0.])
    sphere_bm = euler_maruyama(x0, tn, lambda y: -y/np.linalg.norm(y, ord=2)**2, sphere_orthog_proj, n, seed=seed)
    # Simulate a Brownian motion in the Hyperbolic half-plane
    x0 = np.array([0., 1.])
    hbm = euler_maruyama(x0, tn, lambda y: np.zeros(2), lambda y: np.identity(2) * y[1], n, seed=seed)

    # Plot general SDE
    fig, ((ax1, ax2), ((ax3, ax4))) = plt.subplots(2, 2, figsize=(15, 7))
    ax1.remove()
    ax1 = plt.subplot(2, 2, 1, projection="3d")
    ax1.plot3D(bm3[:, 0], bm3[:, 1], bm3[:, 2], color="black")
    ax1.set_title("Brownian motion")
    ax2.remove()
    ax2 = plt.subplot(2, 2, 2, projection="3d")
    ax2.plot3D(langevin3[:, 0], langevin3[:, 1], langevin3[:, 2], color="black")
    ax2.set_title("Langevin motion with harmonic potential")
    ax3.remove()
    ax3 = plt.subplot(2, 2, 3, projection="3d")
    ax3.plot3D(sphere_bm[:, 0], sphere_bm[:, 1], sphere_bm[:, 2], color="black")
    ax3.set_title("Brownian motion on sphere")
    ax4.plot(hbm[:, 0], hbm[:, 1], color="black")
    ax4.set_title("Hyperbolic BM")
    plt.show()