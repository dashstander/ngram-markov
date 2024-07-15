import torch
from typing import Callable, Tuple


def add(a, b):
    """ Vector addition on the simplex with the Aitchison geometry
    """
    x = torch.multiply(a, b)
    return x / torch.sum(x)


def mul(a, alpha):
    """ Scalar multiplication on the simplex with the Aitchison geometry
    """
    x = torch.pow(a, alpha)
    return x / torch.sum(x)


def inverse(x):
    x = torch.clamp(x, 1e-14, 1e14)
    return torch.reciprocal(x)


def distance(x, y):
    aitch_dist = clr(x) - clr(y)
    return torch.dot(aitch_dist, aitch_dist)


def aitch_basis(dim: int):
    total = dim - 1. + torch.e
    basis = torch.ones((dim, dim))
    i = torch.arange(dim)
    basis[i, i] = torch.e
    return basis / total


def clr_inv(x, dim=-1, initial=None):
    """ The inverse of the CLR transform. Just the softmax, but the code makes more sense with the aliasing
    """
    return torch.nn.functional.softmax(x, dim=dim)


def clr(x, dim: int = -1, keepdim: bool = False):
    """ Centered log ration (clr) transform of a point on the simplex. Takes a point in the canonical basis to 
    """
    log_x = torch.log(x)
    geom_mean = torch.exp(torch.mean(log_x, dim=dim, keepdim=keepdim))
    return torch.log(x / geom_mean)


def aitch_dot(a, b):
    """ Inner product between two elements of the simplex with the Aitchison geometry
    """
    return torch.dot(clr(a), clr(b))


@torch.compile
def ortho_basis_rn(dim):
    def basis_fn(i, j):
        i = i + 1
        j = j + 1
        val = torch.where(j <= i, 1 / i, 
                          torch.where(j == (i + 1), -1., 
                                      torch.zeros_like(i, dtype=torch.float32)))
        return val * torch.sqrt(i / (i + 1))
    
    return torch.stack([torch.stack([basis_fn(torch.tensor(i), torch.tensor(j)) for j in range(dim)]) for i in range(dim-1)])


def ortho_basis_simn(dim: int):
    return torch.vmap(clr_inv)(ortho_basis_rn(dim))


def make_isometric_transforms(dim: int) -> Tuple[Callable]:
    rn_basis = ortho_basis_rn(dim)

    def ilr(x):
        return torch.matmul(rn_basis, clr(x))
    
    def ilr_inv(y):
        return clr_inv(torch.matmul(torch.transpose(rn_basis, 0, 1), y))
    
    return ilr, ilr_inv

@torch.compile
def ilr(x):
    """ x in Sim^D, this function sends it to R^(D-1) according to an orthonormal basis
    """
    d = x.shape[-1]
    ortho = ortho_basis_rn(d)
    return torch.matmul(ortho, clr(x))


@torch.compile
def ilr_inv(y):
    d = y.shape[-1]
    basis = ortho_basis_rn(d + 1)
    return clr_inv(torch.matmul(torch.transpose(basis, 0, 1), y))


def simplex_metric_tensor_inv(x, v):
    def f(x):
        return torch.nn.functional.softmax(x, dim=-1)
    
    _, g_inv = torch.func.jvp(f, x, v)
    return g_inv