from typing import Optional
import torch
import pyro
import pyro.distributions as dist


def Velo2D_S(
    params_dict: dict,
    spliced: Optional[torch.Tensor] = None,
) -> None:
    # unpack dimensions and create plates
    n_cells = params_dict["n_cells"]
    n_genes = params_dict["n_genes"]
    n_basis_S = params_dict["n_basis_s"]
    cell_plate = pyro.plate("cells", n_cells, dim=-1)
    gene_plate = pyro.plate("genes", n_genes, dim=-2)
    prepend_dims = 0 if params_dict["prepend_s"] is None else 1
    spline_plate = pyro.plate("spline", n_basis_S**2 + prepend_dims, dim=-3)

    # coefficients for the spliced spline
    with spline_plate, gene_plate:
        w_s = pyro.sample(
            "w_s",
            dist.Normal(
                torch.full(
                    (n_basis_S**2 + prepend_dims, n_genes, 1), params_dict["w_s_mean"]
                ),
                torch.full(
                    (n_basis_S**2 + prepend_dims, n_genes, 1), params_dict["w_s_scale"]
                ),
            ),
        )
    w_s = w_s.squeeze(-1).T

    # evaluate the spline
    ElogS = torch.mm(w_s, params_dict["design_s"])
    pyro.deterministic("ElogS", ElogS)

    # sample spliced counts
    with gene_plate, cell_plate:
        S = pyro.sample("S", dist.Poisson(torch.exp(ElogS)), obs=spliced)


def Velo2D_U(
    params_dict: dict,
    unspliced: Optional[torch.Tensor] = None,
) -> None:
    # unpack dimensions and create plates
    n_cells = params_dict["n_cells"]
    n_genes = params_dict["n_genes"]
    n_basis_omega = params_dict["n_basis_omega"]
    cell_plate = pyro.plate("cells", n_cells, dim=-1)
    gene_plate = pyro.plate("genes", n_genes, dim=-2)
    omega_plate = pyro.plate(
        "omega_plate", 2 * params_dict["n_basis_omega"] ** 2, dim=-3
    )

    # gene kinetic parameters
    with gene_plate:
        log_beta = pyro.sample(
            "log_beta",
            dist.Normal(
                torch.full((1, n_genes, 1), params_dict["log_beta_mean"]),
                torch.full((1, n_genes, 1), params_dict["log_beta_scale"]),
            ),
        )
        log_gamma = pyro.sample(
            "log_gamma",
            dist.Normal(
                torch.full((1, n_genes, 1), params_dict["log_gamma_mean"]),
                torch.full((1, n_genes, 1), params_dict["log_gamma_scale"]),
            ),
        )
        gamma = torch.exp(log_gamma)
        pyro.deterministic("gamma", gamma)

    # coefficients for the velocity spline
    with omega_plate:
        w_omega = pyro.sample(
            "w_omega",
            dist.Normal(
                torch.full((2 * n_basis_omega**2, 1, 1), params_dict["w_omega_mean"]),
                torch.full((2 * n_basis_omega**2, 1, 1), params_dict["w_omega_scale"]),
            ),
        )
    w_omega = w_omega.squeeze([1, 2]).reshape((2, params_dict["n_basis_omega"] ** 2))

    # evaluate velocity spline
    omega = torch.mm(w_omega, params_dict["design_omega"])
    pyro.deterministic("omega", omega)

    # compute the unspliced
    ElogU = (
        -log_beta
        + params_dict["ElogS"]
        + torch.log(
            torch.relu(
                torch.mm(params_dict["w_s"], params_dict["design_xderiv_s"]) * omega[0]
                + torch.mm(params_dict["w_s"], params_dict["design_yderiv_s"])
                * omega[1]
                + gamma
            )
            + 1e-5
        )
    )
    pyro.deterministic("ElogU", ElogU)

    # sample unspliced counts
    with gene_plate, cell_plate:
        U = pyro.sample("U", dist.Poisson(torch.exp(ElogU)), obs=unspliced)
