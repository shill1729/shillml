from scipy.special import iv as bessel


# Assuming 'indicator' is a Heaviside step function, which in SciPy is 'heaviside'
def indicator(condition):
    return np.heaviside(condition, 0.5)


def dtelegraph(t, x, x0, f, c=1.):
    """
    Exact probability density function solving the telegrapher PDE

    :param t:
    :param x:
    :param x0:
    :param f:
    :param c: speed of light
    :return:
    """
    n = len(t)
    m = len(x)
    v = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            singularPart = f(x[j] - (x0 - c * t[i])) + f(x[j] - (x0 + c * t[i]))
            heaveside = indicator(c * t[i] - abs(x[j] - x0) > 0)

            if heaveside == 1:
                u = np.sqrt((c * t[i]) ** 2 - (x[j] - x0) ** 2)
                i0 = bessel(0, c * u)
                i1 = bessel(1, c * u)
                acPart = heaveside * (c * i0 + (c ** 2) * t[i] * i1 / u)
            else:
                acPart = 0
            v[i, j] = 0.5 * np.exp(-t[i] * c ** 2) * (singularPart + acPart)

    fk = {'t': t, 'x': x, 'u': v}
    return fk


# Example usage:
# Define the function `f` as per your needs, for example:
def example_f(x):
    return np.exp(-x ** 2)


# Then call the function with the required parameters:
# result = function(t_array, x_array, x0, example_f)


# Solve tridiagonal systems fast with the Thomas-algorithm

def tridiag(a, b, c, d):
    """
    Tridiagonal linear system solver using back-substitution, via the
    so-called Thomas algorithm.
    Here 'a' is the lower diagonal, 'b' the diagonal, and 'c' the upper diagonal
    of the tridiagonal matrix A in Ax=d, and this function assumes they are
    vectors/arrays.
    Finally, 'd' is the target vector of some given size greater than 1. Some
    systems will have NaN solutions or infinite.

    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    n = d.size
    cc = np.zeros(n - 1)
    dd = np.zeros(n)
    x = np.zeros(n)
    cc[0] = c[0] / b[0]
    for i in range(1, n - 1, 1):
        cc[i] = c[i] / (b[i] - a[i - 1] * cc[i - 1])
    dd[0] = d[0] / b[0]
    for i in range(1, n, 1):
        dd[i] = (d[i] - a[i - 1] * dd[i - 1]) / (b[i] - a[i - 1] * cc[i - 1])
    x[n - 1] = dd[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - cc[i] * x[i + 1]
    return x


def tridiag_const(a, b, c, d):
    """
    Tridiagonal linear system solver using back-substitution, via the
    so-called Thomas algorithm.
    Here 'a' is the lower diagonal, 'b' the diagonal, and 'c' the upper diagonal
    of the tridiagonal matrix A in Ax=d, and this function assumes they are
    vectors/arrays.
    Finally, 'd' is the target vector of some given size greater than 1. Some
    systems will have NaN solutions or infinite.

    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    n = d.size
    cc = np.zeros(n - 1)
    dd = np.zeros(n)
    x = np.zeros(n)
    cc[0] = c / b
    for i in range(1, n - 1, 1):
        cc[i] = c / (b - a * cc[i - 1])
    dd[0] = d[0] / b
    for i in range(1, n, 1):
        dd[i] = (d[i] - a * dd[i - 1]) / (b - a * cc[i - 1])
    x[n - 1] = dd[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - cc[i] * x[i + 1]
    return x


def indicator(event):
    """ The indicator function of an event (bool)
        Returns 1 if event is true and 0 otherwise. The argument 'event' can be an array of bools
        """
    if type(event) == np.bool_:
        if event:
            return 1
        else:
            return 0
    else:
        y = np.zeros(event.size)
        for i in range(event.size):
            if event[i]:
                y[i] = 1
        return y


# Two classes implementing PDE solvers
# both use finite difference methods (implicit scheme)
# One is for parabolic equations, and one is for a particular
# hyperbolic equation, the telegrapher PDE
class Parabolic:
    def __init__(self, a, b, tn, N, M):
        """ One dimensional parabolic PDE solver via implicit finite-difference scheme.
        """
        self.a = a
        self.b = b
        self.tn = tn
        self.N = N
        self.M = M

    def implicit_scheme(self, g, mu, sigma, f=lambda t, x: 0, r=0, variational=False):
        """

        :param g:
        :param mu:
        :param sigma:
        :param f:
        :param r:
        :param variational:
        :return:
        """
        # Solution matrix
        u = np.zeros((self.N + 1, self.M + 1))
        # Space grid
        x = np.linspace(self.a, self.b, self.M + 1)
        # Time step and space step
        dt = self.tn / self.N
        dx = (self.b - self.a) / self.M

        # I.C.
        u[0, :] = g(x)
        # B.C.
        u[:, 0] = g(x[0])
        u[:, self.M] = g(x[self.M])
        # Time-stepping integration
        for i in range(0, self.N, 1):
            tt = self.tn - (i + 1) * dt
            alpha = (sigma(tt, x) ** 2) / (2 * (dx ** 2)) - (mu(tt, x)) / (2 * dx)
            beta = -r - (sigma(tt, x) / dx) ** 2
            delta = (sigma(tt, x) ** 2) / (2 * (dx ** 2)) + (mu(tt, x)) / (2 * dx)
            ff = f(tt, x[1:self.M])
            a = -dt * alpha[1:self.M]
            b = 1 - dt * beta[0:self.M]
            c = -dt * delta[0:(self.M - 1)]
            # Setting up the target vector
            d = np.zeros(self.M - 1)
            d[0] = alpha[0] * u[0, 0]
            d[self.M - 2] = delta[self.M - 2] * u[0, self.M]
            di = u[i, 1:self.M] + dt * (d + ff)
            u[i + 1, 1:self.M] = tridiag(a, b, c, di)
            if variational:
                u[i + 1, :] = 0.5 * (u[i + 1, :] + g(x) + np.abs(u[i + 1, :] - g(x)))
                # for j in range(0, self.M+1):
                #    u[i+1, j] = np.max(a = np.array([u[i+1, j], g(x[j])]))
        return u

    def implicit_scheme_const(self, g, mu, sigma, f=lambda t, x: 0, r=0, variational=False):
        """

        :param g:
        :param mu:
        :param sigma:
        :param f:
        :param r:
        :param variational:
        :return:
        """
        # Solution matrix
        u = np.zeros((self.N + 1, self.M + 1))
        # Space grid
        x = np.linspace(self.a, self.b, self.M + 1)
        # Time step and space step
        dt = self.tn / self.N
        dx = (self.b - self.a) / self.M

        # I.C.
        u[0, :] = g(x)
        # B.C.
        u[:, 0] = g(x[0])
        u[:, self.M] = g(x[self.M])
        alpha = (sigma ** 2) / (2 * (dx ** 2)) - (mu) / (2 * dx)
        beta = -r - (sigma / dx) ** 2
        delta = (sigma ** 2) / (2 * (dx ** 2)) + (mu) / (2 * dx)
        a = -dt * alpha
        b = 1 - dt * beta
        c = -dt * delta
        # Time-stepping integration
        for i in range(0, self.N, 1):
            tt = self.tn - (i + 1) * dt
            ff = f(tt, x[1:self.M])
            # Setting up the target vector
            d = np.zeros(self.M - 1)
            d[0] = alpha * u[0, 0]
            d[self.M - 2] = delta * u[0, self.M]
            di = u[i, 1:self.M] + dt * (d + ff)
            u[i + 1, 1:self.M] = tridiag_const(a, b, c, di)
            # Adjustment made for variational inequalities
            if variational:
                u[i + 1, :] = 0.5 * (u[i + 1, :] + g(x) + np.abs(u[i + 1, :] - g(x)))
        return u


class Telegraph:
    """

    """
    def __init__(self, x0, T, c):
        """

        :param x0:
        :param T:
        :param c:
        """
        self.x0 = x0
        self.T = T
        self.c = c
        self._N = 500
        self._M = 500
        self._a = 1.5 * (self.x0 - self.c * self.T)
        self._b = 1.5 * (self.x0 + self.c * self.T)
        self._dx = (self._b - self._a) / self._M
        self._dt = self.T / self._M

    def set_resolution(self, time_res, space_res):
        """

        :param time_res:
        :param space_res:
        :return:
        """
        self._N = time_res
        self._M = space_res

    def generate(self, t, N=10):
        """
        Use this method for generating multiple samples at a single instant of time of Kac's
        telegraph process.

        The method generates a random number of Poisson jumps in the given time interval,
        then their jump-times and conditional on this, we have a nice expression for Kac's random time
        as a simple alternating sum of the waiting times.

        :param t:
        :param N:
        :return:
        """
        tau = np.zeros(N)
        for i in range(N):

            n = np.random.poisson(self.c ** 2, size=1)
            if n > 0:

                w = np.random.exponential(1 / self.c ** 2, size=n)
                tn = np.sum(w)
                k = np.arange(0, n)
                tau[i] = (t - tn) * (-1) ** n + np.sum(w * (-1) ** k)
            else:
                tau[i] = t
        return tau

    def telegrapher_mc1(self, t, x, f, N):
        """
        Solve the one dimensional telegrapher PDE, in this case a Fokker-Planck setting for Kac's random
        time process. The method here is a Monte-Carlo scheme that averages the wave-equation's
        solution over realizations of Kac's random time.

        :param t:
        :param x:
        :param f:
        :param N:
        :return:
        """
        n = t.shape[0]
        m = x.shape[0]
        p = np.zeros((n, m))
        for i in range(0, n):
            tau = self.generate(t[i], N)
            for j in range(0, m):
                u = 0.5 * (f(x[j] + self.c * tau) + f(x[j] - self.c * tau))
                p[i, j] = np.mean(u)
        return p

    def initial_condition(self, x, epsilon=0.001):
        """

        :param epsilon:
        :param x:
        :return:
        """
        # return indicator(np.abs(x - self.x0) < self._dx) / self._dx
        return np.exp(-(x-self.x0)**2/(2*epsilon))/(np.sqrt(2*np.pi)*epsilon)

    def _initial_velocity(self, x):
        """

        :param x:
        :return:
        """
        return np.zeros(x.size)

    def implicit_scheme(self, epsilon=0.001):
        """

        :return:
        """
        # Coefficients defining the tridiagonal linear-system
        k1 = 1 + 1 / (self._dt * self.c ** 2)
        k2 = -1 / (2 * self._dt * self.c ** 2)
        alpha = 1 - k2 + self._dt / self._dx ** 2
        beta = -self._dt / (2 * self._dx ** 2)
        x = np.linspace(self._a, self._b, self._M + 1)
        u = np.zeros((self._N + 1, self._M + 1))
        # IC
        u[0, :] = self.initial_condition(x, epsilon)
        u[1,] = self._initial_velocity(x) * self._dt + u[0,]
        # BC
        u[:, 0] = self.initial_condition(x[0])
        u[:, self._M] = self.initial_condition(x[self._M])
        # BC are zero, default
        for i in range(2, self._N + 1, 1):
            d = k1 * u[i - 1, 2:self._M] + k2 * u[i - 2, 2:self._M]
            u1 = tridiag_const(beta, alpha, beta, d)
            u[i, 2:self._M] = u1
        return u

    def plot_pde(self, ax, epsilon=0.001):
        """
        Plot the PDE solution surface for a telegrapher PDE.

        :param ax: the axis
        :param epsilon: size for approximating Dirac-Delta by Gaussian
        :return:
        """
        # Computing solution grid to PDE problem
        u = self.implicit_scheme(epsilon)
        # 3D mesh
        time = np.linspace(0, self.T, self._N + 1)
        space = np.linspace(self._a, self._b, self._M + 1)
        # For 3D plotting
        time, space = np.meshgrid(time, space)

        # Plotting solution surface
        sc = ax.plot_surface(time, space, u.transpose(), cmap="viridis")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title("Finite-difference solution")
        fig.colorbar(sc, ax=ax, label="p(t,x)")


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Variance for Gaussian approximation to Dirac-delta function
    epsilon = 0.0001
    n = 500
    m = 500
    N = 500
    c = 0.307
    tn = 1.
    x0 = 0
    b = c * tn * 1.2
    t = np.linspace(0, tn, n + 1)
    x = np.linspace(x0 - b, x0 + b, m + 1)
    # Instantiate Telegraph class
    tt = Telegraph(x0, tn, c)
    tt.set_resolution(n, m)
    tt._a = x0 - b
    tt._b = x0 + b

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    tt.plot_pde(ax, epsilon)
    plt.show()

    def f(x):
        return tt.initial_condition(x, epsilon)

    # Monte-Carlo solution
    p = tt.telegrapher_mc1(t, x, f, N)
    time, space = np.meshgrid(t, x)

    # Plotting solution surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.plot_surface(time, space, p.transpose(), cmap="viridis")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Monte-Carlo solution")
    fig.colorbar(sc, ax=ax, label="p(t,x)")
    plt.show()

    # Exact solution
    p = dtelegraph(t, x, x0, f, c)
    fig = plt.figure()
    # Plotting solution surface
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.plot_surface(time, space, p["u"].transpose(), cmap="viridis")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Exact solution")
    fig.colorbar(sc, ax=ax, label="p(t,x)")
    plt.show()
