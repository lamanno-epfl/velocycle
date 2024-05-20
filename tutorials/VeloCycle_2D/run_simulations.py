import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import tqdm
import yaml
import scanpy as sc
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoNormal
from pyro.optim import Adam

from simulate_2D import generate_adata
from model_2d import Velo2D_S, Velo2D_U
from splines_torch import torch_spline_basis_2d, torch_spline_basis_2d_der, derivative


def train_velo2d(
    S: torch.Tensor,
    U: torch.Tensor,
    params_dict: dict,
    n_steps_S: int = 1000,
    n_steps_U: int = 1000,
    lr_S: float = 1e-2,
    lr_U: float = 1e-2,
) -> tuple:
    pyro.clear_param_store()
    model_S = Velo2D_S
    guide_S = AutoNormal(Velo2D_S)
    svi = SVI(Velo2D_S, guide_S, Adam({"lr": lr_S}), loss=Trace_ELBO())
    losses_S = []
    for i in tqdm.trange(n_steps_S):
        loss = svi.step(params_dict, S)
        losses_S.append(loss)

    params_dict["w_s"] = (
        pyro.get_param_store()["AutoNormal.locs.w_s"].squeeze(-1).T.detach()
    )
    params_dict["ElogS"] = torch.mm(params_dict["w_s"], params_dict["design_s"])
    params_S = pyro.get_param_store().get_state()
    pyro.clear_param_store()

    model_U = Velo2D_U
    guide_U = AutoNormal(Velo2D_U)
    svi = SVI(Velo2D_U, guide_U, Adam({"lr": lr_U}), loss=Trace_ELBO())
    losses_U = []
    for i in tqdm.trange(n_steps_U):
        loss = svi.step(params_dict, U)
        losses_U.append(loss)
    params_U = pyro.get_param_store().get_state()

    return model_S, guide_S, params_S, losses_S, model_U, guide_U, params_U, losses_U


def extract_velo2d_results(
    params_dict: dict,
    params_S: dict,
    params_U: dict,
) -> dict:
    result = {}

    # save results of velocity learning
    pyro.get_param_store().set_state(params_U)
    result["w_omega"] = (
        pyro.get_param_store()["AutoNormal.locs.w_omega"]
        .squeeze([1, 2])
        .reshape((2, params_dict["n_basis_omega"] ** 2))
        .detach()
        .numpy()
    )
    result["omega"] = np.dot(result["w_omega"], params_dict["design_omega"])
    result["log_beta"] = (
        pyro.get_param_store()["AutoNormal.locs.log_beta"]
        .squeeze([0, 2])
        .detach()
        .numpy()
    )
    result["log_gamma"] = (
        pyro.get_param_store()["AutoNormal.locs.log_gamma"]
        .squeeze([0, 2])
        .detach()
        .numpy()
    )
    result["ratio"] = result["log_gamma"] - result["log_beta"]
    params_U = pyro.get_param_store().get_state()
    pyro.clear_param_store()

    # save results of manifold learning
    pyro.get_param_store().set_state(params_S)
    result["w_s"] = (
        pyro.get_param_store()["AutoNormal.locs.w_s"].squeeze(-1).T.detach().numpy()
    )
    result["ElogS"] = np.dot(result["w_s"], params_dict["design_s"])
    result["ElogU"] = (
        -result["log_beta"][:, None]
        + result["ElogS"]
        + np.log(
            np.maximum(
                np.dot(result["w_s"], params_dict["design_xderiv_s"])
                * result["omega"][0]
                + np.dot(result["w_s"], params_dict["design_yderiv_s"])
                * result["omega"][1]
                + np.exp(result["log_gamma"][:, None]),
                1e-5,
            )
        )
    )

    return result


def prepare_params_dict(
    phi: Union[np.ndarray, torch.Tensor],
    n_basis_s: int,
    n_basis_omega: int,
):
    knots_inner = torch.linspace(0, 10, n_basis_s - 2)
    knots_left = torch.full((3,), 0)
    knots_right = torch.full((3,), 10)
    knots_s = torch.concatenate([knots_left, knots_inner, knots_right])
    knots_sder, c_sder, _ = derivative(knots_s, 3, np.eye(n_basis_s))
    c_sder = torch.tensor(c_sder)
    design_s = torch_spline_basis_2d(
        phi[0],
        phi[1],
        t=knots_s,
        k=3,
        c=None,
        prepend=1,
    ).T
    design_yderiv_s, design_xderiv_s = torch_spline_basis_2d_der(
        phi[0],
        phi[1],
        t=knots_s,
        tder=knots_sder,
        k=3,
        kder=2,
        c=c_sder,
        prepend=1,
    )
    design_xderiv_s, design_yderiv_s = design_xderiv_s.T, design_yderiv_s.T

    knots_inner = torch.linspace(0, 10, n_basis_omega - 2)
    knots_left = torch.full((3,), 0)
    knots_right = torch.full((3,), 10)
    knots_omega = torch.concatenate([knots_left, knots_inner, knots_right])
    design_omega = torch_spline_basis_2d(
        phi[0],
        phi[1],
        t=knots_omega,
        k=3,
        c=None,
        prepend=None,
    ).T

    params_dict = {
        "n_cells": 3000,
        "n_genes": 300,
        "phi": phi,
        "n_basis_s": n_basis_s,
        "n_basis_omega": n_basis_omega,
        "design_s": design_s,
        "design_xderiv_s": design_xderiv_s,
        "design_yderiv_s": design_yderiv_s,
        "design_omega": design_omega,
        "log_beta_mean": 2.0,
        "log_beta_scale": 1.0,
        "log_gamma_mean": 0.25,
        "log_gamma_scale": 0.1,
        "w_s_mean": 0.0,
        "w_s_scale": 5.0,
        "w_omega_mean": 0.0,
        "w_omega_scale": 1.0,
        "prepend_s": 1,
    }

    return params_dict


if __name__ == "__main__":
    np.random.seed(0)
    seeds = np.random.randint(0, 1000, size=10)

    w_omega = np.load("w_omega.npy")
    correlations = {
        "omega": [],
        "spliced_exp": [],
        "unspliced_exp": [],
        "spliced": [],
        "unspliced": [],
        "log_beta": [],
        "log_gamma": [],
        "ratio": [],
    }
    for i, seed in enumerate(seeds):
        folder = f"results/sim_{seed}"
        os.makedirs(folder, exist_ok=True)

        # generate data
        adata, spliced_exp, unspliced_exp = generate_adata(
            n_cells=3000,
            n_genes=300,
            w_omega=w_omega,
            seed=seed,
            plot=True,
            plot_filename=f"{folder}/sim.png",
        )
        adata.obs["phi2"] = adata.obsm["phi"][:, 1]
        sc.pp.pca(adata)
        sc.pl.pca(adata, color="phi2", layer="spliced", show=False)
        plt.savefig(f"{folder}/pca.png")
        plt.close()

        S = torch.tensor(adata.layers["spliced"]).T.float()
        U = torch.tensor(adata.layers["unspliced"]).T.float()
        phi = torch.tensor(adata.obsm["phi"]).T.float()
        params_dict = prepare_params_dict(phi, 6, 6)

        # train model
        pyro.set_rng_seed(seed)
        (
            model_S,
            guide_S,
            params_S,
            losses_S,
            model_U,
            guide_U,
            params_U,
            losses_U,
        ) = train_velo2d(
            S,
            U,
            params_dict,
            n_steps_S=int(5e3),
            n_steps_U=int(5e3),
            lr_S=1e-2,
            lr_U=1e-3,
        )
        plt.plot(losses_S, label="S")
        plt.plot(losses_U, label="U")
        plt.legend()
        plt.savefig(f"{folder}/losses.png")
        plt.close()

        # save generated data and results
        result = extract_velo2d_results(params_dict, params_S, params_U)
        adata.write(f"{folder}/adata.h5ad")
        np.save(f"{folder}/spliced_exp.npy", spliced_exp)
        np.save(f"{folder}/unspliced_exp.npy", unspliced_exp)
        np.save(f"{folder}/result.npy", result)

        # compute correlation with ground truth
        correlations["omega"].append(
            [
                np.corrcoef(adata.obsm["omega"].T[0], result["omega"][0])[0, 1].item(),
                np.corrcoef(adata.obsm["omega"].T[1], result["omega"][1])[0, 1].item(),
            ]
        )
        correlations["spliced_exp"].append(
            np.corrcoef(spliced_exp.flatten(), np.exp(result["ElogS"]).flatten())[
                0, 1
            ].item()
        )
        correlations["unspliced_exp"].append(
            np.corrcoef(unspliced_exp.flatten(), np.exp(result["ElogU"]).flatten())[
                0, 1
            ].item()
        )
        correlations["spliced"].append(
            np.corrcoef(
                adata.layers["spliced"].T.flatten(), np.exp(result["ElogS"]).flatten()
            )[0, 1].item()
        )
        correlations["unspliced"].append(
            np.corrcoef(
                adata.layers["unspliced"].T.flatten(), np.exp(result["ElogU"]).flatten()
            )[0, 1].item()
        )
        correlations["log_beta"].append(
            np.corrcoef(np.log(adata.var["beta"]), result["log_beta"])[0, 1].item()
        )
        correlations["log_gamma"].append(
            np.corrcoef(np.log(adata.var["gamma"]), result["log_gamma"])[0, 1].item()
        )
        correlations["ratio"].append(
            np.corrcoef(
                np.log(adata.var["gamma"]) - np.log(adata.var["beta"]),
                result["ratio"],
            )[0, 1].item()
        )

    with open(f"results/correlations.yaml", "w") as f:
        yaml.dump(correlations, f)
