# This module has sympy functions for dealing with SDEs (on manifolds).
import sympy as sp
import numpy as np


def matrix_divergence(a: sp.Matrix, x: sp.Matrix):
    """
    Compute the matrix divergence, i.e. the Euclidean divergence applied row-wise to a matrix
    :param a:
    :param x:
    :return:
    """
    n, m = a.shape
    d = sp.zeros(n, 1)
    for i in range(n):
        for j in range(m):
            d[i] += sp.diff(a[i, j], x[j])
    return d


def hessian(f: sp.Matrix, x: sp.Matrix, return_grad=False):
    """
    Compute the Hessian of a scalar field.
    :param f:
    :param x:
    :param return_grad: whether to return (hessian, gradient) or just hessian
    :return:
    """
    grad_f = f.jacobian(x)
    hess_f = grad_f.jacobian(x)
    if return_grad:
        return hess_f, grad_f.T
    else:
        return hess_f


def metric_tensor(chart, x):
    """
    Compute the metric tensor given a diffeomorphic chart from low to high dimension.

    :param chart:
    :param x:
    :return:
    """
    j = sp.simplify(chart.jacobian(x))
    g = j.T * j
    return sp.simplify(g)


def manifold_divergence(a: sp.Matrix, p: sp.Matrix, volume_measure):
    """
    Compute the manifold divergence of a matrix (row-wise) at a point given the volume measure
    :param volume_measure:
    :param a:
    :param p:
    :return:
    """

    drift = sp.simplify(matrix_divergence(sp.simplify(volume_measure * a), p))
    drift = sp.simplify(drift / volume_measure)
    return drift


def embedded_bm_coefficients(f: sp.Matrix, p: sp.Matrix):
    """
    Compute the drift and diffusion coefficients of a Riemannian Brownian motion embedded in some
    large Euclidean space.

    :param f: sympy matrix, the function in the implicit equation f(x)=0 defining the manifold globally
    :param p: sympy matrix, the point (x,y,z) etc
    :return: tuple of (drift, P, H), of the drift vector, orthogonal projection matrix, and orthonormal frame matrix
    """
    D = p.shape[0]  # Embedded dimension
    K = f.shape[0]  # Codimension
    # Compute the Jacobian of the implicit function
    Df = f.jacobian(p)
    # Orthonormalize $ N^T N = Df^T A^{-1} Df, A=Df Df^T
    A = Df * Df.T
    B = A.inv()
    P = sp.eye(D, D) - Df.T * B * Df
    P = sp.simplify(P)
    # Compute the orthonormal frame: H = first d columns of U sqrt(S), P=USV^T
    P = sp.Matrix(P)
    U, S, V = P.singular_value_decomposition()
    H = U * sp.sqrt(S)

    d = D - K
    H = sp.simplify(H[:, 0:d])
    # Compute the drift: the mean-curvature times the normal vector
    C = sp.simplify(B.cholesky(hermitian=False))
    N = sp.simplify(C * Df)
    mean_curvature = -matrix_divergence(N, p) / 2
    # Second order term for intersections of hypersurfaces
    q = sp.Matrix.zeros(K, 1)
    if K > 1:
        for i in range(K):
            Dn = sp.simplify(N[i, :].jacobian(p))
            q[i] = sp.simplify(sp.Trace(N * Dn * N.T))
        q = q / 2
    drift = sp.simplify(N.T * (mean_curvature + q))
    return drift, P, H


def local_bm_coefficients(g: sp.Matrix, p):
    """
    Compute the SDE coefficients of a Brownian motion in a local chart of a manifold

    :param g: metric tensor
    :param p: point
    :return:
    """
    # 1. Compute the diffusion coefficient
    ginv = sp.simplify(g.inv())
    diffusion = sp.simplify(ginv.cholesky(hermitian=False))
    # 2. Compute the drift
    detg = g.det()
    sqrt_detg = sp.sqrt(detg)
    manifold_div = manifold_divergence(ginv, p, sqrt_detg)
    drift = sp.simplify(manifold_div / 2)
    return drift, diffusion


def infinitesimal_generator(f, p: sp.Matrix, drift: sp.Matrix, diffusion: sp.Matrix):
    """
    Compute the infinitesimal generator of a SDE

    :param f: test function
    :param p: point to evaluate at
    :param drift: drift coefficient
    :param diffusion: diffusion coefficient

    :return:
    """
    h = sp.Matrix([f])
    hess_f, grad_f = hessian(h, p, return_grad=True)
    first_order_term = drift.T * grad_f
    first_order_term = sp.simplify(first_order_term)[0, 0]
    cov = sp.simplify(diffusion * diffusion.T)
    quadratic_variation_term = sp.simplify(cov * hess_f)
    second_order_term = quadratic_variation_term.trace() / 2
    inf_gen = first_order_term + second_order_term
    inf_gen = sp.simplify(inf_gen)
    return inf_gen


def adjoint_generator(f, p: sp.Matrix, drift: sp.Matrix, diffusion: sp.Matrix):
    """

    :param f:
    :param p:
    :param drift:
    :param diffusion:
    :return:
    """
    Sigma = sp.simplify(diffusion * diffusion.T)
    second_order_term = sp.simplify(matrix_divergence(Sigma * f, p) / 2)
    flux = sp.simplify(-drift * f + second_order_term)
    # Since matrix divergence is implemented row-wise.
    adjoint = sp.simplify(matrix_divergence(flux.T, p))
    return adjoint


def sympy_to_numpy_coefficients(mu, sigma, p):
    """
    Convert sympy SDE coefficients to numpy SDE coefficients as functions of p

    :param mu:
    :param sigma:
    :param p:
    :return:
    """
    d = mu.shape[0]
    return lambda x: sp.lambdify([p], mu)(x).reshape(d), sp.lambdify([p], sigma)


# Creating surfaces via the parameterizations
def surf_param(coord, chart, grid, aux=None, p=None):
    """ Compute a mesh of a surface via parameterization. The argument
    'grid' must be a tuple of arrays returned from 'np.mgrid' which the user
    must supply themselves, since boundaries and resolutions are use-case dependent.
    The tuple returned can be unpacked and passed to plot_surface

    (Parameters):
    coord: sympy object defining parameters
    chart: sympy object defining the coordinate transformation
    grid: tuple of the arrays returned from np.mgrid[...]
    aux: sympy Matrix for auxiliary parameters in the metric tensor
    p: numpy array for the numerical values of any auxiliary parameters in the equations

    returns tuple (x,y,z), (x,y) or (x)
    """
    d = len(grid)
    m = grid[0].shape[0]
    N = chart.shape[0]
    if aux is None:
        chart_np = sp.lambdify([coord], chart)
    else:
        chart_np = sp.lambdify([coord, aux], chart)

    xx = np.zeros((N, grid[0].shape[0], grid[0].shape[1]))
    for i in range(m):
        for j in range(m):
            w = np.zeros(d)
            for l in range(d):
                w[l] = grid[l][i, j]
            for l in range(N):
                if aux is None:
                    xx[l][i, j] = chart_np(w)[l, 0]
                else:
                    xx[l][i, j] = chart_np(w, p)[l, 0]
    if N == 3:
        x = xx[0]
        y = xx[1]
        z = xx[2]
        return x, y, z
    elif N == 2:
        x = xx[0]
        y = xx[1]
        return x, y
    else:
        return xx


def lift_path(xt, f, m=3):
    """

    :param xt: local path
    :param f: diffeomorphism lifting to the manifold
    :param m: higher dimension
    :return:
    """
    ntime = xt.shape[0] - 1
    # We need to lift the motion back to the ambient space
    yt = np.zeros((ntime + 1, m))
    for i in range(ntime + 1):
        yt[i, :] = f(xt[i, :]).reshape(m)
    return yt


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shillml.sdes.solvers import euler_maruyama

    x, y = sp.symbols("x y", real=True)
    f = sp.Matrix([x ** 2 + x * y ** 2])
    p = sp.Matrix([x, y])
    print(hessian(f, p))

    # sympy input
    theta, phi = sp.symbols("theta phi", real=True, positive=True)
    x = sp.sin(theta) * sp.cos(phi)
    y = sp.sin(theta) * sp.sin(phi)
    z = sp.cos(theta)
    xi = sp.Matrix([x, y, z])
    coord = sp.Matrix([theta, phi])
    g = metric_tensor(xi, coord)

    # Path input
    x0 = np.array([1.5, 3.14])
    tn = 1
    seed = 17
    n = 9000

    mu, sigma = local_bm_coefficients(g, coord)
    f = sp.Function("f")(*coord)
    inf_gen = infinitesimal_generator(f, coord, mu, sigma)
    adj_gen = adjoint_generator(f, coord, mu, sigma)
    mu_np, sigma_np = sympy_to_numpy_coefficients(mu, sigma, coord)
    harmonic_test = infinitesimal_generator(sp.sin(theta), coord, mu, sigma)
    fokker_planck_test = adjoint_generator(sp.sin(theta), coord, mu, sigma)
    print("Metric tensor")
    print(g)
    print("Local drift")
    print(mu)
    print("Local diffusion")
    print(sigma)
    print("Infinitesimal generator")
    print(inf_gen)
    print("Harmonic test")
    print(harmonic_test)
    print("Fokker Planck RHS")
    print(adj_gen)
    print("Fokker Planck Test")
    print(fokker_planck_test)

    xt = euler_maruyama(x0, tn, mu_np, sigma_np, n, seed=seed)
    yt = lift_path(xt, sp.lambdify([coord], xi))
    # Surface grid
    grid1 = np.linspace(0, np.pi, 100)
    grid2 = np.linspace(0, 2 * np.pi, 100)
    grid = np.meshgrid(grid1, grid2, indexing="ij")
    x1, x2, x3 = surf_param(coord, xi, grid)

    fig = plt.figure()
    ax = plt.subplot(projection="3d")
    ax.plot3D(yt[:, 0], yt[:, 1], yt[:, 2], color="black")
    ax.plot_surface(x1, x2, x3, cmap="viridis", alpha=0.5)
    plt.show()
