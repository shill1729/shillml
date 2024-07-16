from typing import List

import sympy as sp
from sympy import Matrix, MutableDenseNDimArray

from symcalc import matrix_divergence


class RiemannianManifold:
    """
    A class representing a Riemannian manifold.

    This class provides methods to compute various geometric quantities and
    perform calculations on a Riemannian manifold defined by local coordinates
    and a chart.

    Attributes:
        local_coordinates (Matrix): The local coordinates of the manifold.
        chart (Matrix): The chart defining the manifold.
    """

    def __init__(self, local_coordinates: Matrix, chart: Matrix):
        """
        Initialize the RiemannianManifold.

        Args:
            local_coordinates (Matrix): The local coordinates of the manifold.
            chart (Matrix): The chart defining the manifold.
        """
        self.local_coordinates = local_coordinates
        self.chart = chart

    def chart_jacobian(self) -> Matrix:
        """
        Compute the Jacobian of the chart.

        Returns:
            Matrix: The Jacobian matrix of the chart.
        """
        return sp.simplify(self.chart.jacobian(self.local_coordinates))

    def metric_tensor(self) -> Matrix:
        """
        Compute the metric tensor of the manifold.

        Returns:
            Matrix: The metric tensor.
        """
        j = self.chart_jacobian()
        g = j.T * j
        return sp.simplify(g)

    def volume_density(self) -> sp.Expr:
        """
        Compute the volume density of the manifold.

        Returns:
            sp.Expr: The volume density.
        """
        g = self.metric_tensor()
        return sp.simplify(sp.sqrt(sp.simplify(g.det())))

    def g_orthonormal_frame(self, method: str = "pow") -> Matrix:
        """
        Compute the g-orthonormal frame of the manifold.

        Args:
            method (str): The method to use for computation. Either "pow" or "svd".

        Returns:
            Matrix: The g-orthonormal frame.

        Raises:
            ValueError: If an invalid method is specified.
        """
        g = self.metric_tensor()
        g_inv = g.inv()
        if method == "pow":
            return sp.simplify(g_inv.pow(1/2))
        elif method == "svd":
            u, s, v = sp.Matrix(g_inv).singular_value_decomposition()
            sqrt_g_inv = u * sp.sqrt(s) * v
            return sp.simplify(sqrt_g_inv)
        else:
            raise ValueError("argument 'method' must be 'pow' or 'svd'.")

    def orthonormal_frame(self, method: str = "pow") -> Matrix:
        """
        Compute the orthonormal frame of the manifold.

        Args:
            method (str): The method to use for computation. Either "pow" or "svd".

        Returns:
            Matrix: The orthonormal frame.
        """
        j = self.chart_jacobian()
        e = self.g_orthonormal_frame(method)
        return sp.simplify(j * e)

    def orthogonal_projection(self, method: str = "pow") -> Matrix:
        """
        Compute the orthogonal projection of the manifold.

        Args:
            method (str): The method to use for computation. Either "pow" or "svd".

        Returns:
            Matrix: The orthogonal projection.
        """
        h = self.orthonormal_frame(method)
        return sp.simplify(h * h.T)

    def manifold_divergence(self, f: Matrix) -> sp.Expr:
        """
        Compute the manifold divergence of a vector field.

        Args:
            f (Matrix): The vector field.

        Returns:
            sp.Expr: The manifold divergence.
        """
        vol_den = self.volume_density()
        scaled_field = sp.simplify(vol_den*f)
        manifold_div = matrix_divergence(scaled_field, self.local_coordinates)/vol_den
        return sp.simplify(manifold_div)

    def local_bm_drift(self) -> Matrix:
        """
        Compute the local Brownian motion drift.

        Returns:
            Matrix: The local Brownian motion drift.
        """
        g_inv = sp.simplify(self.metric_tensor().inv())
        return sp.simplify(self.manifold_divergence(g_inv)/2)

    def local_bm_diffusion(self, method: str = "pow") -> Matrix:
        """
        Compute the local Brownian motion diffusion.

        Args:
            method (str): The method to use for computation. Either "pow" or "svd".

        Returns:
            Matrix: The local Brownian motion diffusion.
        """
        return self.g_orthonormal_frame(method)

    def christoffel_symbols(self) -> MutableDenseNDimArray:
        """
        Compute the Christoffel symbols of the manifold.

        Returns:
            MutableDenseNDimArray: The Christoffel symbols.
        """
        g = self.metric_tensor()
        g_inv = sp.simplify(g.inv())
        n = len(self.local_coordinates)
        christoffel = sp.MutableDenseNDimArray.zeros(n, n, n)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    term = 0
                    for l in range(n):
                        term += g_inv[k, l] * (sp.diff(g[i, l], self.local_coordinates[j]) +
                                               sp.diff(g[j, l], self.local_coordinates[i]) -
                                               sp.diff(g[i, j], self.local_coordinates[l]))
                    christoffel[k, i, j] = sp.simplify(term / 2)

        return christoffel

    def riemann_curvature_tensor(self) -> MutableDenseNDimArray:
        """
        Compute the Riemann curvature tensor of the manifold.

        Returns:
            MutableDenseNDimArray: The Riemann curvature tensor.
        """
        gamma = self.christoffel_symbols()
        n = len(self.local_coordinates)
        R = sp.MutableDenseNDimArray.zeros(n, n, n, n)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        term1 = sp.diff(gamma[l, i, k], self.local_coordinates[j])
                        term2 = sp.diff(gamma[l, i, j], self.local_coordinates[k])
                        term3 = sum(gamma[m, i, k] * gamma[l, m, j] for m in range(n))
                        term4 = sum(gamma[m, i, j] * gamma[l, m, k] for m in range(n))
                        R[l, i, j, k] = sp.simplify(term1 - term2 + term3 - term4)

        return R

    def ricci_curvature_tensor(self) -> MutableDenseNDimArray:
        """
        Compute the Ricci curvature tensor of the manifold.

        Returns:
            MutableDenseNDimArray: The Ricci curvature tensor.
        """
        R = self.riemann_curvature_tensor()
        n = len(self.local_coordinates)
        Ric = sp.MutableDenseNDimArray.zeros(n, n)

        for i in range(n):
            for j in range(n):
                Ric[i, j] = sum(R[k, i, k, j] for k in range(n))

        return sp.simplify(Ric)

    def scalar_curvature(self) -> sp.Expr:
        """
        Compute the scalar curvature of the manifold.

        Returns:
            sp.Expr: The scalar curvature.
        """
        Ric = self.ricci_curvature_tensor()
        g_inv = self.metric_tensor().inv()
        return sp.simplify(sum(Ric[i, j] * g_inv[i, j] for i in range(len(self.local_coordinates)) for j in
                               range(len(self.local_coordinates))))

    def sectional_curvature(self, u: Matrix, v: Matrix) -> sp.Expr:
        """
        Compute the sectional curvature of the manifold.

        Args:
            u (Matrix): First vector.
            v (Matrix): Second vector.

        Returns:
            sp.Expr: The sectional curvature.
        """
        R = self.riemann_curvature_tensor()
        g = self.metric_tensor()
        n = len(self.local_coordinates)
        num = sum(R[i, j, k, l] * u[i] * v[j] * u[k] * v[l]
                  for i in range(n) for j in range(n)
                  for k in range(n) for l in range(n))
        den = (sum(g[i, j] * u[i] * u[j] for i in range(n) for j in range(n)) *
               sum(g[i, j] * v[i] * v[j] for i in range(n) for j in range(n)) -
               (sum(g[i, j] * u[i] * v[j] for i in range(n) for j in range(n))) ** 2)
        return sp.simplify(num / den)

    def levi_civita_connection(self, vector_field: Matrix) -> MutableDenseNDimArray:
        """
        Compute the Levi-Civita connection of a vector field.

        Args:
            vector_field (Matrix): The vector field.

        Returns:
            MutableDenseNDimArray: The Levi-Civita connection.
        """
        gamma = self.christoffel_symbols()
        n = len(self.local_coordinates)
        result = sp.MutableDenseNDimArray.zeros(n)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i] += gamma[i, j, k] * vector_field[j] * vector_field[k]
        return sp.simplify(result)

    def geodesic_equations(self) -> List[sp.Expr]:
        """
        Compute the geodesic equations of the manifold.

        Returns:
            List[sp.Expr]: The geodesic equations.
        """
        gamma = self.christoffel_symbols()
        n = len(self.local_coordinates)
        t = sp.Symbol('t')
        x = sp.Matrix([sp.Function(f'x{i}')(t) for i in range(n)])
        dx = x.diff(t)
        d2x = dx.diff(t)

        equations = []
        for i in range(n):
            eq = d2x[i] + sum(gamma[i, j, k] * dx[j] * dx[k]
                              for j in range(n) for k in range(n))
            equations.append(eq)

        return equations

    def lie_bracket(self, X: Matrix, Y: Matrix) -> MutableDenseNDimArray:
        """
        Compute the Lie bracket of two vector fields.

        Args:
            X (Matrix): First vector field.
            Y (Matrix): Second vector field.

        Returns:
            MutableDenseNDimArray: The Lie bracket.
        """
        n = len(self.local_coordinates)
        result = sp.MutableDenseNDimArray.zeros(n)

        for i in range(n):
            result[i] = sum(X[j] * Y[i].diff(self.local_coordinates[j]) -
                            Y[j] * X[i].diff(self.local_coordinates[j])
                            for j in range(n))
        return sp.simplify(result)

    def covariant_derivative(self, X: Matrix, Y: Matrix) -> MutableDenseNDimArray:
        """
        Compute the covariant derivative of a vector field with respect to another.

        Args:
            X (Matrix): The vector field to differentiate with respect to.
            Y (Matrix): The vector field to differentiate.

        Returns:
            MutableDenseNDimArray: The covariant derivative.
        """
        n = len(self.local_coordinates)
        gamma = self.christoffel_symbols()
        result = sp.MutableDenseNDimArray.zeros(n)

        for i in range(n):
            result[i] = sum(X[j] * Y[i].diff(self.local_coordinates[j])
                            for j in range(n))
            result[i] += sum(gamma[i, j, k] * X[j] * Y[k]
                             for j in range(n) for k in range(n))
        return sp.simplify(result)

    def parallel_transport(self, v: Matrix, curve: Matrix) -> List[sp.Expr]:
        """
        Compute the parallel transport equations along a curve.

        Args:
            v (Matrix): The vector to transport.
            curve (Matrix): The curve to transport along.

        Returns:
            List[sp.Expr]: The parallel transport equations.
        """
        t = sp.Symbol('t')
        n = len(self.local_coordinates)
        gamma = self.christoffel_symbols()
        equations = []

        for i in range(n):
            eq = v[i].diff(t) + sum(gamma[i, j, k] * curve.diff(t)[j] * v[k]
                                    for j in range(n) for k in range(n))
            equations.append(eq)
        return equations


if __name__ == "__main__":
    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    # Sphere
    chart = sp.Matrix([sp.sin(u) * sp.cos(v), sp.sin(u) * sp.sin(v), sp.cos(u)])
    # chart = sp.Matrix([u, v, u**2 + v**2])
    man = RiemannianManifold(local_coord, chart)

    print("Sphere")
    print("\nChart Jacobian:")
    print(man.chart_jacobian())

    print("\nMetric tensor:")
    print(man.metric_tensor())

    print("\ng-orthonormal frame:")
    print(man.g_orthonormal_frame())

    print("\nOrthonormal frame:")
    print(man.orthonormal_frame())

    print("\nOrthogonal projection:")
    print(man.orthogonal_projection())

    print("\nVolume density:")
    print(man.volume_density())

    print("\nLocal drift:")
    print(man.local_bm_drift())

    print("\nChristoffel Symbols:")
    print(man.christoffel_symbols())

    print("\nRiemannian curvature tensor:")
    print(man.riemann_curvature_tensor())

    print("\nRicci curvature tensor:")
    print(man.ricci_curvature_tensor())

    print("\nScalar curvature")
    print(man.scalar_curvature())

    print("\nSectional curvature:")
    e1 = sp.Matrix([1, 0])
    e2 = sp.Matrix([0, 1])
    print(man.sectional_curvature(e1, e2))

    print("\nGeodesic equations:")
    print(man.geodesic_equations())

    print("\nLie bracket of coordinate vector fields:")
    X = sp.Matrix([1, 0])
    Y = sp.Matrix([0, 1])
    print(man.lie_bracket(X, Y))

    print("\nCovariant derivative of e1 with respect to e2:")
    print(man.covariant_derivative(e2, e1))

    print("\nParallel transport equations along u-coordinate curve:")
    t = sp.Symbol('t')
    curve = sp.Matrix([t, 0])
    v = sp.Matrix([sp.Function('v1')(t), sp.Function('v2')(t)])
    print(man.parallel_transport(v, curve))
