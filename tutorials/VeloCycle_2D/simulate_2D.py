import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
from typing import Optional, Tuple
from splines_torch import (
    torch_spline_basis_2d,
    torch_spline_basis_2d_der,
    derivative,
)


def _generate_cells(
    n_cells_before: int,
    n_cells_after: int,
    phi_lower: float,
    phi_upper: float,
    x_branching: float,
    y_initial: float,
    sd_min: float,
    sd_max: float,
) -> np.ndarray:
    """
    Generate the positions of the cells forming a 2D branching structure.
    """
    _phi_lower = phi_lower + 0.1 * (phi_upper - phi_lower)
    _phi_upper = phi_upper - 0.1 * (phi_upper - phi_lower)

    sd = np.concatenate(
        [
            np.linspace(sd_min, sd_max, n_cells_before),
            np.linspace(sd_max, sd_min, n_cells_after),
        ]
    )

    slope_upper = (_phi_lower + (_phi_upper - _phi_lower) - y_initial) / (
        _phi_upper - x_branching
    )
    intercept_upper = y_initial - slope_upper * x_branching
    slope_lower = -slope_upper
    intercept_lower = y_initial - slope_lower * x_branching

    phi = np.zeros((2, n_cells_before + n_cells_after))
    phi[0, :n_cells_before] = np.linspace(_phi_lower, x_branching, n_cells_before)
    phi[0, n_cells_before:] = np.linspace(x_branching, _phi_upper, n_cells_after)
    phi[1, :n_cells_before] = 5
    upper = np.random.rand(n_cells_after) > 0.5
    phi[1, n_cells_before:] = np.where(
        upper,
        slope_upper * phi[0, n_cells_before:] + intercept_upper,
        slope_lower * phi[0, n_cells_before:] + intercept_lower,
    )

    # add noise
    phi += np.random.normal(0, sd, (2, n_cells_before + n_cells_after))
    phi[phi < phi_lower] = phi_lower
    phi[phi > phi_upper] = phi_upper

    return phi


def _generate_spline_basis(
    phi: np.ndarray,
    phi_lower: float,
    phi_upper: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the spline basis for the given points.
    """
    n_basis = 6
    degree = 3
    knots_left = np.full((3,), phi_lower)
    knots_right = np.full((3,), phi_upper)

    knots_inner = np.linspace(phi_lower, phi_upper, n_basis - 2)
    knots_s = np.concatenate([knots_left, knots_inner, knots_right])
    knots_sder, c_sder, _ = derivative(knots_s, degree, np.eye(n_basis))
    design_s = torch_spline_basis_2d(
        phi[0],
        phi[1],
        t=knots_s,
        k=degree,
        c=None,
        prepend=1,
    ).T.numpy()
    design_yderiv_s, design_xderiv_s = torch_spline_basis_2d_der(
        phi[0],
        phi[1],
        t=knots_s,
        tder=knots_sder,
        k=degree,
        kder=degree - 1,
        c=c_sder,
        prepend=1,
    )
    design_xderiv_s, design_yderiv_s = (
        design_xderiv_s.T.numpy(),
        design_yderiv_s.T.numpy(),
    )

    knots_inner = np.linspace(phi_lower, phi_upper, n_basis - 2)
    knots_omega = np.concatenate([knots_left, knots_inner, knots_right])
    design_omega = torch_spline_basis_2d(
        phi[0],
        phi[1],
        t=knots_omega,
        k=degree,
        c=None,
        prepend=None,
    ).T.numpy()

    return design_s, design_yderiv_s, design_xderiv_s, design_omega


def _generate_velocity(
    design_omega: np.ndarray,
    w_omega: Optional[np.ndarray] = None,
):
    """
    Generate a velocity field for the given points.
    """
    if w_omega is None:
        w_omega = np.random.normal(0, 1, (2, design_omega.shape[0]))
    omega = np.dot(w_omega, design_omega)
    return omega


def _generate_gene_expression(
    design_s: np.ndarray,
    design_xderiv_s: np.ndarray,
    design_yderiv_s: np.ndarray,
    omega: np.ndarray,
    log_beta_mean: float = 2.0,
    log_beta_scale: float = 0.7,
    log_gamma_mean: float = 0.25,
    log_gamma_scale: float = 0.1,
    module: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Generate single gene expression using splines.

    Depending on the module, one of the following expression patterns is generated:
    0 - lower in upper branch
    1 - lower in lower branch
    2 - higher in upper branch
    3 - higher in lower branch
    4 - higher in both branches
    5 - lower in both branches
    6 - higher in upper, lower in lower
    7 - lower in upper, higher in lower
    """
    passes_checks = False
    while not passes_checks:
        # gene kinetic parameters
        log_beta = np.random.normal(log_beta_mean, log_beta_scale)
        log_gamma = np.random.normal(log_gamma_mean, log_gamma_scale)
        gamma = np.exp(log_gamma)

        # coefficients for the spliced spline
        w_s_means = np.zeros(6)
        sd_root = 1.0
        sd_middle = 0.2
        sd_tip = 0.6
        if module == 0:
            w_s_means[[1, 2, 4]] = 1.0
            w_s_means[[3, 5]] = 0.0
        elif module == 1:
            w_s_means[[2, 4]] = 0.0
            w_s_means[[1, 3, 5]] = 1.0
        elif module == 2:
            w_s_means[[1, 2, 4]] = 0.0
            w_s_means[[3, 5]] = 1.0
        elif module == 3:
            w_s_means[[1, 3, 5]] = 0.0
            w_s_means[[2, 4]] = 1.0
        elif module == 4:
            w_s_means[1] = 0.0
            w_s_means[[2, 3, 4, 5]] = 1.0
        elif module == 5:
            w_s_means[1] = 1.0
            w_s_means[[2, 3, 4, 5]] = 0.0
        elif module == 6:
            w_s_means[1] = 0.5
            w_s_means[[2, 4]] = 0.0
            w_s_means[[3, 5]] = 1.0
        elif module == 7:
            w_s_means[1] = 0.5
            w_s_means[[2, 4]] = 1.0
            w_s_means[[3, 5]] = 0.0
        w_s = np.random.normal(
            w_s_means, [sd_root, sd_middle, sd_middle, sd_middle, sd_tip, sd_tip]
        )
        w_s = np.concatenate(
            (
                np.array([0.0, 0.0, w_s[0], w_s[0], 0.0, 0.0]),
                np.array([0.0, 0.0, w_s[0], w_s[0], 0.0, 0.0]),
                np.array([0.0, 0.0, w_s[1], w_s[1], 0.0, 0.0]),
                np.array([0.0, w_s[2] / 2, w_s[2], w_s[3], w_s[3] / 2, 0.0]),
                np.array([w_s[4], w_s[4], w_s[2] / 2, w_s[3] / 2, w_s[5], w_s[5]]),
                np.concatenate((np.full(2, w_s[4]), [0.0, 0.0], np.full(2, w_s[5]))),
            )
        )
        intercept = np.random.normal(2.0, 1.0)
        w_s = np.concatenate([np.array([intercept]), w_s])
        w_s = w_s[None, :]
        inner_term = (
            np.dot(w_s, design_xderiv_s) * omega[0]
            + np.dot(w_s, design_yderiv_s) * omega[1]
            + gamma
        )
        if (inner_term > 0).all():
            passes_checks = True

        ElogS = np.dot(w_s, design_s).squeeze(0)
        ElogU = -log_beta + ElogS + np.log(np.maximum(inner_term, 0) + 1e-5)

        spliced = np.exp(ElogS)
        unspliced = np.exp(ElogU)

        if np.any((spliced > 1000) | (unspliced > 1000)):
            passes_checks = False
            continue

        if (spliced > 10).sum() < 0.2 * len(spliced):
            passes_checks = False
            continue

    return spliced, unspliced, np.exp(log_beta), gamma, w_s


def generate_adata(
    n_cells: int = 1000,
    n_genes: int = 200,
    phi_lower: float = 0.0,
    phi_upper: float = 10.0,
    x_branching: float = 5.0,
    y_initial: float = 5.0,
    cell_sd_min: float = 0.2,
    cell_sd_max: float = 0.6,
    w_omega: Optional[np.ndarray] = None,
    seed: int = 0,
    plot: bool = False,
    plot_filename: Optional[str] = None,
) -> ad.AnnData:
    """
    Generate AnnData for a 2D branching single-cell dataset.

    Parameters
    ----------
    n_cells: int
        number of cells
    n_genes: int
        number of genes
    phi_lower: float
        lower bound for the cell positions
    phi_upper: float
        upper bound for the cell positions
    x_branching: float
        x-coordinate of the branching point
    y_initial: float
        initial y-coordinate of the cells
    cell_sd_min: float
        min standard deviation of cell positions (near root and tips)
    cell_sd_max: float
        max standard deviation of cell positions (near branching point)
    w_omega: np.ndarray
        velocity field weights
    seed: int
        random seed
    plot: bool
        whether to plot the generated data
    plot_filename: str
        filename for the plot (if None, show the plot)
    """
    np.random.seed(seed)

    # divide the cells into two parts
    n_cells_before = n_cells // 3
    n_cells_after = n_cells - n_cells_before

    # generate the cell positions
    phi = _generate_cells(
        n_cells_before,
        n_cells_after,
        phi_lower,
        phi_upper,
        x_branching,
        y_initial,
        cell_sd_min,
        cell_sd_max,
    )
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        ax = axs[0]
        ax.scatter(phi[0], phi[1])
        ax.set_title("Cell positions")
        ax.set_xlabel("cell_x")
        ax.set_ylabel("cell_y")

    # generate the spline basis
    design_s, design_yderiv_s, design_xderiv_s, design_omega = _generate_spline_basis(
        phi,
        phi_lower,
        phi_upper,
    )

    # generate the velocity field
    omega = _generate_velocity(design_omega, w_omega)
    if plot:
        ax = axs[1]
        ax.quiver(phi[0], phi[1], omega[0], omega[1])
        ax.set_title("Velocity field")
        ax.set_xlabel("cell_x")
        ax.set_ylabel("cell_y")

    # generate the gene expression
    betas, gammas, weights = [], [], []
    for i in range(n_genes):
        _spliced, _unspliced, beta, gamma, w_s = _generate_gene_expression(
            design_s,
            design_xderiv_s,
            design_yderiv_s,
            omega,
            module=i % 8,
        )
        betas.append(beta)
        gammas.append(gamma)
        weights.append(w_s)
        if i == 0:
            spliced, unspliced = _spliced, _unspliced
            continue
        spliced = np.vstack([spliced, _spliced])
        unspliced = np.vstack([unspliced, _unspliced])

    # plot gene expression example
    if plot:
        i = np.random.randint(0, n_genes)
        ax = axs[2]
        ax.scatter(phi[0], phi[1], c=spliced[i], s=10)
        plt.colorbar(ax.collections[0], ax=ax, orientation="vertical")
        ax.set_xlabel("cell_x")
        ax.set_ylabel("log(counts)")
        ax.set_title("Example of spliced counts")
        fig.tight_layout()

        if plot_filename is not None:
            plt.savefig(plot_filename)
            plt.close()
        else:
            plt.show()

    # create anndata object
    spliced_exp = spliced.copy()
    unspliced_exp = unspliced.copy()
    spliced = np.random.poisson(spliced).T
    unspliced = np.random.poisson(unspliced).T
    adata = ad.AnnData(
        X=spliced + unspliced, layers={"spliced": spliced, "unspliced": unspliced}
    )
    adata.obsm["phi"] = phi.T
    adata.obsm["omega"] = omega.T
    adata.var["beta"] = betas
    adata.var["gamma"] = gammas

    return adata, spliced_exp, unspliced_exp
