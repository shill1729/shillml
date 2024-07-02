import sympy as sp


def matrix_divergence(a : sp.Matrix, x : sp.Matrix):
    """
    Compute the matrix divergence, i.e. the Euclidean divergence applied row-wise to a matrix
    :param a:
    :param x:
    :return:
    """
    n, m = a.shape
    d = sp.zeros(n,1)
    for i in range(n):
        for j in range(m):
            d[i] += sp.simplify(sp.diff(a[i, j], x[j]))
    return sp.simplify(d)


def hessian(f : sp.Matrix, x : sp.Matrix, return_grad=False):
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


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)
    f = sp.Matrix([x**2+x*y**2])
    p = sp.Matrix([x, y])
    print(hessian(f, p))

