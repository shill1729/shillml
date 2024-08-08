import sympy as sp
import numpy as np


class Surfaces:
    def __init__(self):
        """
        Initialize the Surfaces class with predefined symbols and parameters.

        Attributes:
        -----------
        u : sympy.Symbol
            Symbol representing the first parameter of the surface equations.
        v : sympy.Symbol
            Symbol representing the second parameter of the surface equations.
        R : float
            Radius of the torus.
        r : float
            Inner radius of the torus.
        c1 : float
            Coefficient used in the quartic, paraboloid, hyperbolic_paraboloid, and ellipsoid surfaces.
        c2 : float
            Coefficient used in the quartic, paraboloid, hyperbolic_paraboloid, and ellipsoid surfaces.
        sd : float
            Standard deviation used for Gaussian bump.
        surface_bounds : dict
            Dictionary containing the parameter bounds for each surface type.
        """
        self.u, self.v = sp.symbols('u v', real=True)
        self.R = 2
        self.r = 1
        self.c1 = 5
        self.c2 = 5
        self.sd = 5
        self.surface_bounds = {
            'quartic': [(-1, 1), (-1, 1)],
            'paraboloid': [(-1, 1), (-1, 1)],
            'sphere': [(0, np.pi), (0, 2 * np.pi)],
            'torus': [(0, 2 * np.pi), (0, 2 * np.pi)],
            'gaussian_bump': [(-3, 3), (-3, 3)],
            'hyperbolic_paraboloid': [(-2, 2), (-2, 2)],
            'hyperboloid': [(0, 2 * np.pi), (-2, 2)],
            'cylinder': [(0, 2 * np.pi), (-2, 2)],
            'mobius_strip': [(0, 2 * np.pi), (-1, 1)],
            'helicoid': [(-np.pi, np.pi), (-1, 1)],
            'dinis_surface': [(0, 2 * np.pi), (0.1, 0.9 * np.pi)],
            'cone': [(0, 1), (0, 2 * np.pi)],
            'hyperboloid_one_sheet': [(-1, 1), (0, 2 * np.pi)],
            'hyperboloid_two_sheets': [(1, 2), (0, 2 * np.pi)],
            'ellipsoid': [(0, np.pi), (0, 2 * np.pi)],
            'klein_bottle': [(0, np.pi), (0, 2 * np.pi)],
            'enneper_surface': [(-2, 2), (-2, 2)]
        }

    def quartic(self):
        """
        Define a quartic surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the quartic surface.
        """
        return sp.Matrix([self.u, self.v, (self.u / self.c1) ** 4 - (self.v / self.c2) ** 2])

    def paraboloid(self):
        """
        Define a paraboloid surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the paraboloid surface.
        """
        return sp.Matrix([self.u, self.v, (self.u / self.c1) ** 2 + (self.v / self.c2) ** 2])

    def hyperbolic_paraboloid(self):
        """
        Define a hyperbolic paraboloid surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the hyperbolic paraboloid surface.
        """
        return sp.Matrix([self.u, self.v, (self.v / self.c2) ** 2 - (self.u / self.c1) ** 2])

    def hyperboloid(self):
        """
        Define a hyperboloid surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the hyperboloid surface.
        """
        return sp.Matrix([sp.cosh(self.v) * sp.cos(self.u), sp.cosh(self.v) * sp.sin(self.u), sp.sinh(self.v)])

    def sphere(self):
        """
        Define a sphere surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the sphere surface.
        """
        return sp.Matrix([sp.sin(self.u) * sp.cos(self.v), sp.sin(self.u) * sp.sin(self.v), sp.cos(self.u)])

    def torus(self):
        """
        Define a torus surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the torus surface.
        """
        return sp.Matrix([(self.R + self.r * sp.cos(self.u)) * sp.cos(self.v),
                          (self.R + self.r * sp.cos(self.u)) * sp.sin(self.v),
                          self.r * sp.sin(self.u)])

    def gaussian_bump(self):
        """
        Define a Gaussian bump surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the Gaussian bump surface.
        """
        return sp.Matrix([self.u, self.v, sp.exp(-(self.u ** 2 + self.v ** 2) / 2) / sp.sqrt(2 * sp.pi)])

    def cylinder(self):
        """
        Define a cylinder surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the cylinder surface.
        """
        return sp.Matrix([sp.cos(self.u), sp.sin(self.u), self.v])

    def mobius_strip(self):
        """
        Define a Möbius strip surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the Möbius strip surface.
        """
        return sp.Matrix([(1 + (self.v * sp.cos(self.u / 2)) / 2) * sp.cos(self.u),
                          (1 + (self.v * sp.cos(self.u / 2)) / 2) * sp.sin(self.u),
                          (self.v / 2) * sp.sin(self.u / 2)])

    def helicoid(self):
        """
        Define a helicoid surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the helicoid surface.
        """
        return sp.Matrix([self.v * sp.cos(3 * self.u), self.v * sp.sin(3 * self.u), self.u])

    def dinis_surface(self):
        """
        Define a Dini's surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of Dini's surface.
        """
        return sp.Matrix([sp.cos(self.u) * sp.sin(self.v),
                          sp.sin(self.u) * sp.cos(self.v),
                          self.u + sp.log(sp.tan(self.v / 2)) + sp.cos(self.v)])

    def cone(self):
        """
        Define a cone surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the cone surface.
        """
        return sp.Matrix([self.u * sp.cos(self.v), self.u * sp.sin(self.v), self.u])

    def hyperboloid_one_sheet(self):
        """
        Define a hyperboloid of one sheet surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the hyperboloid of one sheet surface.
        """
        return sp.Matrix([sp.cosh(self.u) * sp.cos(self.v),
                          sp.cosh(self.u) * sp.sin(self.v),
                          sp.sinh(self.u)])

    def hyperboloid_two_sheets(self):
        """
        Define a hyperboloid of two sheets surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the hyperboloid of two sheets surface.
        """
        return sp.Matrix([sp.sinh(self.u) * sp.cos(self.v),
                          sp.sinh(self.u) * sp.sin(self.v),
                          sp.cosh(self.u)])

    def ellipsoid(self):
        """
        Define an ellipsoid surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the ellipsoid surface.
        """
        return sp.Matrix([self.c1 * sp.sin(self.u) * sp.cos(self.v),
                          self.c2 * sp.sin(self.u) * sp.sin(self.v),
                          self.r * sp.cos(self.u)])

    def klein_bottle(self):
        """
        Define a Klein bottle surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the Klein bottle surface.
        """
        return sp.Matrix([(sp.cos(self.u) * (1 + sp.sin(self.u)) + self.v * sp.cos(self.u) / 2) * sp.cos(self.u),
                          (sp.cos(self.u) * (1 + sp.sin(self.u)) + self.v * sp.cos(self.u) / 2) * sp.sin(self.u),
                          self.v * sp.sin(self.u) / 2])

    def enneper_surface(self):
        """
        Define an Enneper surface.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the Enneper surface.
        """
        return sp.Matrix([self.u - (self.u ** 3) / 3 + self.u * self.v ** 2,
                          self.v - (self.v ** 3) / 3 + self.v * self.u ** 2,
                          self.u ** 2 - self.v ** 2])


# Example of usage
if __name__ == "__main__":
    generator = Surfaces()
    print(generator.torus())
    print(generator.surface_bounds['torus'])
