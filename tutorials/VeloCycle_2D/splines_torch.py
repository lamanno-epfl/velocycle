import numpy as np
import torch
from scipy.interpolate import splder
from typing import Union, Optional


def spline_prep(lower_bound=0, upper_bound=1, df=6, degree=3):
    """Returns the nodes and degree of of a spline

    Returns
    ------
    t: np.ndarray
        The nodes
    k: int
        The degree
    """
    order = degree + 1
    n_inner_knots = df - order
    inner_knots = np.linspace(lower_bound, upper_bound, n_inner_knots + 2)[1:-1]
    all_knots = np.concatenate(
        ([lower_bound] * order, inner_knots, [upper_bound] * order)
    )
    t = all_knots
    k = degree
    return t, k


def torch_B(x, k, i, t):

    if k == 0:
        res = torch.ones(len(x), len(i), dtype=x.dtype)
        res[~((t[i] <= x) & (x <= t[i + 1]))] = 0.0
        return res

    c1 = torch.zeros(len(x), len(i), dtype=x.dtype)
    c2 = torch.zeros(len(x), len(i), dtype=x.dtype)
    bool_ = ~(t[i + k] == t[i])
    c1[:, ~bool_] = 0
    c1[:, bool_] = (
        (x - t[i][bool_]) / (t[i + k] - t[i])[bool_] * torch_B(x, k - 1, i, t)[:, bool_]
    )

    bool_ = ~(t[i + k + 1] == t[i + 1])
    c2[:, ~bool_] = 0
    c2[:, bool_] = (
        (t[i + k + 1][bool_] - x)
        / (t[i + k + 1] - t[i + 1])[bool_]
        * torch_B(x, k - 1, i + 1, t)[:, bool_]
    )

    return c1 + c2


def tvect_B(x, k, i, t):
    """Fully vectorial, but less efficient version
    The 1e-16 is required otherwise backpropagation fails!
    """
    if k == 0:
        return torch.where(
            (t[i] <= x) & (x <= t[i + 1]),
            torch.tensor([1.0], dtype=x.dtype),
            torch.tensor([0.0], dtype=x.dtype),
        )
    c1 = torch.where(
        t[i + k] == t[i],
        torch.tensor([0.0], dtype=x.dtype),
        (x - t[i]) / (t[i + k] - t[i] + 1e-16) * tvect_B(x, k - 1, i, t),
    )

    c2 = torch.where(
        t[i + k + 1] == t[i + 1],
        torch.tensor([0.0], dtype=x.dtype),
        (t[i + k + 1] - x)
        / (t[i + k + 1] - t[i + 1] + 1e-16)
        * tvect_B(x, k - 1, i + 1, t),
    )
    return c1 + c2


def derivative(t, k, c=None, nu=1):
    """Returns the nodes and degree and the weights for
    the spline basis that can be used to represent
    the derivative of any function expressed as spline expansion

    Returns
    ------
    tder: np.ndarray
        The nodes of the derivative
    c: np.ndarray, 2-D
        The coefficients to right-multiply to the
        spline basis so to keep the coefficient of
        a function spline-representation
    kder: int
        The degree of the derivative
    """
    if c is None:
        n = len(t) - k - 1
        c = np.eye(n, dtype=t.dtype)
    ct = len(t) - len(c)
    if ct > 0:
        c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
    tck = splder((t, c, k), nu)
    return tck


def torch_spline_basis(
    x: Union[torch.Tensor, np.ndarray],
    t: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    c: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prepend: Optional[int] = None,
) -> torch.Tensor:
    """Usage note
    c is only needed for derivatives
    Example:
    basis = torch_spline_basis(x, t, k)
    tder, c, kder = derivative(t, k, nu=1)
    der_basis = torch_spline_basis(x, tder, kder, c)
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(t, np.ndarray):
        t = torch.tensor(t)
    if c is not None and isinstance(c, np.ndarray):
        c = torch.tensor(c)

    n = len(t) - k - 1
    D = torch_B(x[:, None], k, torch.arange(n), t.type(x.dtype))
    if c is None:
        pass
    else:
        D = D @ c.type(D.dtype)[:n, :]

    if prepend == 0:
        return torch.column_stack([torch.zeros(D.shape[0]), D])
    elif prepend == 1:
        return torch.column_stack([torch.ones(D.shape[0]), D])
    elif not (prepend is None):
        return torch.column_stack([torch.full(D.shape[0], prepend), D])
    else:
        return D


def tvect_spline_basis(x, t, k, c=None, prepend=None):
    n = len(t) - k - 1
    D = tvect_B(x[:, None], k, torch.arange(n), t.type(x.dtype))
    if c is None:
        pass
    else:
        D = D @ c.type(D.dtype)[:n, :]

    if prepend == 0:
        return torch.column_stack([torch.zeros(D.shape[0]), D])
    elif prepend == 1:
        return torch.column_stack([torch.ones(D.shape[0]), D])
    elif not (prepend is None):
        return torch.column_stack([torch.full(D.shape[0], prepend), D])
    else:
        return D


def torch_spline_basis_2d(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    t: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    c: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prepend: Optional[int] = None,
) -> torch.Tensor:
    """Usage
    D = torch_spline_basis_2d(x, ty, t, k)
    """
    Dx = torch_spline_basis(x, t, k)
    Dy = torch_spline_basis(y, t, k)
    Dxy = Dy.repeat((1, Dx.shape[1])) * Dx.repeat_interleave(Dy.shape[1], dim=1)
    if prepend == 0:
        return torch.column_stack([torch.zeros(Dxy.shape[0]), Dxy])
    elif prepend == 1:
        return torch.column_stack([torch.ones(Dxy.shape[0]), Dxy])
    elif not (prepend is None):
        return torch.column_stack([torch.full(Dxy.shape[0], prepend), Dxy])
    else:
        return Dxy


def torch_spline_basis_2d_der(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    t: Union[torch.Tensor, np.ndarray],
    tder: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    kder: int = 2,
    c: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prepend: Optional[int] = None,
):
    """Usage
    tder, c, kder = derivative(t, k, nu=1)
    D_dy, D_dx = torch_spline_basis_2d_der(x, y, t, k, tder, kder, c)
    """
    Dx = torch_spline_basis(x, t, k)
    Dy = torch_spline_basis(y, t, k)
    Dxdx = torch_spline_basis(x, tder, kder, c)
    Dydy = torch_spline_basis(y, tder, kder, c)
    Dxydy = Dydy.repeat((1, Dx.shape[1])) * Dx.repeat_interleave(Dydy.shape[1], dim=1)
    Dxydx = Dy.repeat((1, Dxdx.shape[1])) * Dxdx.repeat_interleave(Dy.shape[1], dim=1)
    if prepend == 0:
        return (
            torch.column_stack([torch.zeros(Dxydy.shape[0]), Dxydy]),
            torch.column_stack([torch.zeros(Dxydx.shape[0]), Dxydx]),
        )
    elif prepend == 1:
        return (
            torch.column_stack([torch.zeros(Dxydy.shape[0]), Dxydy]),
            torch.column_stack([torch.zeros(Dxydx.shape[0]), Dxydx]),
        )
    elif not (prepend is None):
        return (
            torch.column_stack([torch.full(Dxydy.shape[0], prepend), Dxydy]),
            torch.column_stack([torch.full(Dxydx.shape[0], prepend), Dxydx]),
        )
    else:
        return Dxydy, Dxydx
