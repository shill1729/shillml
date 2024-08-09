import sympy as sp


class SDECoefficients:
    def __init__(self):
        """
        Initialize the SDECoefficients class with predefined symbols u and v.

        Attributes:
        -----------
        u : sympy.Symbol
            Symbol representing the first variable of the drift and diffusion equations.
        v : sympy.Symbol
            Symbol representing the second variable of the drift and diffusion equations.
        """
        self.u, self.v = sp.symbols(names="u v", real=True)

    @staticmethod
    def drift_zero():
        """
        Define the zero drift vector.

        Returns:
        --------
        sympy.Matrix
            Zero drift vector [0, 0].
        """
        return sp.Matrix([0, 0])

    def drift_circular(self):
        """
        Define the circular drift vector.

        Returns:
        --------
        sympy.Matrix
            Circular drift vector [-sin(u) + v^2 - 1, exp(-v)].
        """
        return sp.Matrix([-sp.sin(self.u) + self.v ** 2 - 1, sp.exp(-self.v)])

    def drift_harmonic_potential(self):
        """
        Define the harmonic potential drift vector.

        Returns:
        --------
        sympy.Matrix
            Harmonic potential drift vector [-0.5 * u, -0.5 * v].
        """
        return -0.5 * sp.Matrix([self.u, self.v])

    def drift_morse_potential(self):
        """
        Define the Morse potential drift vector.

        Returns:
        --------
        sympy.Matrix
            Morse potential drift vector [2 * u * exp(-u^2 - v^2), 2 * v * exp(-u^2 - v^2)].
        """
        return -sp.Matrix([-2 * self.u * sp.exp(-self.u ** 2 - self.v ** 2),
                           -2 * self.v * sp.exp(-self.u ** 2 - self.v ** 2)])

    def drift_double_well_potential(self):
        """
        Define the double well potential drift vector.

        Returns:
        --------
        sympy.Matrix
            Double well potential drift vector [4 * u * (u^2 - 1), 2 * v].
        """
        return -sp.Matrix([4 * self.u * (self.u ** 2 - 1), 2 * self.v])

    def drift_lennard_jones_potential(self):
        """
        Define the Lennard-Jones potential drift vector.

        Returns:
        --------
        sympy.Matrix
            Lennard-Jones potential drift vector
            [12 * u / (u^2 + v^2)^7 - 24 * u / (u^2 + v^2)^13,
             12 * v / (u^2 + v^2)^7 - 24 * v / (u^2 + v^2)^13].
        """
        return -sp.Matrix([12 * self.u / (self.u ** 2 + self.v ** 2) ** 7 -
                           24 * self.u / (self.u ** 2 + self.v ** 2) ** 13,
                           12 * self.v / (self.u ** 2 + self.v ** 2) ** 7 -
                           24 * self.v / (self.u ** 2 + self.v ** 2) ** 13])

    def diffusion_identity(self):
        """
        Define the identity diffusion matrix.

        Returns:
        --------
        sympy.Matrix
            Identity diffusion matrix [[1, 0], [0, 1]].
        """
        return sp.Matrix([[1, 0], [0, 1]])

    def diffusion_diagonal(self):
        """
        Define the diagonal diffusion matrix.

        Returns:
        --------
        sympy.Matrix
            Diagonal diffusion matrix [[u, 0], [0, v]] * 0.25.
        """
        return sp.Matrix([[self.u, 0], [0, self.v]]) * 0.25

    def diffusion_circular(self):
        """
        Define the circular diffusion matrix.

        Returns:
        --------
        sympy.Matrix
            Circular diffusion matrix [[5 + 0.5 * cos(u), v], [0, 0.2 + 0.1 * sin(v)]].
        """
        return sp.Matrix([[5 + 0.5 * sp.cos(self.u), self.v],
                          [0, 0.2 + 0.1 * sp.sin(self.v)]])


# Example of usage
if __name__ == "__main__":
    sde_coeffs = SDECoefficients()
    print(sde_coeffs.drift_harmonic_potential())
    print(sde_coeffs.diffusion_identity())
