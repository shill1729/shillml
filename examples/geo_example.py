"""
    This module computes various differential geometric objects given a parameterization of the form
    (u, v, f(u,v)) and simulates a RBM in the local chart and ambient space.
"""
if __name__ == "__main__":
    import sympy as sp
    import numpy as np
    from sympy import sin, cos
    from shillml.diffgeo import RiemannianManifold

    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([sin(u)*cos(v), sin(u)*sin(v), cos(u)])
    man = RiemannianManifold(local_coord, chart)
    tn = 3.5
    npaths = 10
    ntime = 15000
    x0 = np.array([1., 1.])
    print("Simulating paths...")
    # local_paths, global_paths = man.simulate_rbm(x0, tn, ntime, npaths)
    # man.plot_rbm(local_paths, global_paths)

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