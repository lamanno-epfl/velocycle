#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from collections import namedtuple
import scipy
import copy

# Relative imports
from .phases import Phases
from .cycle import Cycle, reorder
from .angularspeed import AngularSpeed
from .phase_inference_model import *
from .velocity_inference_model import *
from .phase_inference_guide import *
from .velocity_inference_guide import *


def filter_shared_genes(cycle, data, filter_type="intersection"):
    """A function to take a unified subset of genes between a Cycle object and an AnnData object

    Arguments
    ---------
    cycle: object, class Cycle
        Cycle object to subset
    data: object, class AnnData
        AnnData object to subset
    filter_type: str, the type of gene filtering behavior to carry out
        the following are currently supported:
        - intersection
        - union
        for union, there is the additional requirement that all genes are already present in AnnData object

    Returns
    -------
    new_cycle: object, class Cycle
        Subsetted Cycle object
    new_data: object, class AnnData
        Subsetted AnnData object
        
    """
    cycle_genes = set(cycle.genes)
    data_genes = set(data.var.index)
    if filter_type=="intersection": # keep only genes in both Cycle and AnnData object
        keep_genes = np.array(list(cycle_genes & data_genes))
        keep_genes = keep_genes[np.argsort(keep_genes)]
        new_data = data[:, keep_genes].copy()
        new_cycle = Cycle.from_array(means_array=cycle.means[keep_genes], 
                                     stds_array=cycle.stds[keep_genes])
    elif filter_type=="union": # keep all genes in either Cycle and AnnData object
        if len(cycle_genes - data_genes)>0:
            raise Exception('Gene features detected in Cycle object cannot be found in AnnData object')
        keep_genes = np.array(list(cycle_genes | data_genes))
        keep_genes = keep_genes[np.argsort(keep_genes)]
        new_data = data[:, keep_genes].copy()
        new_cycle = Cycle.from_array(means_array=cycle.means, 
                                     stds_array=cycle.stds)
        new_cycle.extend(gene_names=np.array(list(data_genes-cycle_genes)))
        new_cycle = reorder(new_cycle, keep_genes)
    else:
        raise Exception("Error: invalid argument for filter_type")
    return new_cycle, new_data

def make_design_matrix(anndata, ids="batch"):
    """A function to create a generic design matrix from cell identifiers
    
    For example:
    input, cell_identifiers = [1,1,1,2,2,2]
    output, design_matrix: [[1,1,1,0,0,0][0,0,0,1,1,1]]

    Arguments
    ---------
    cell_identifiers: torch tensor
        List of cells identifiers, should be of type torch.int
        
    Returns
    -------
    design_matrix: torch tensor
        Design matrix of shape (num_cells, num_unique_identifiers)
        
    """
    from collections import defaultdict
    
    temp = defaultdict(lambda: len(temp))
    cell_identifiers = torch.tensor(np.array([temp[ele] for ele in np.array(anndata.obs[ids])])+1)
    
    if ids not in anndata.obs.columns:
        raise ValueError(f"{ids=} is not a valid entry anndata.obs")
    
    design_matrix = torch.hstack([(cell_identifiers == v).type(torch.int64).reshape(len(cell_identifiers), 
                                                              1) for i, v in enumerate(torch.unique(cell_identifiers))])
    return design_matrix

def normalize_total(anndata):
    anndata.obs["n_scounts"] = anndata.layers["spliced"].toarray().sum(1)
    anndata.obs["n_ucounts"] = anndata.layers["unspliced"].toarray().sum(1)
    norm_factors = np.mean(anndata.obs["n_scounts"]) / np.array(anndata.obs["n_scounts"])
    anndata.layers["S_sz"] = (norm_factors * anndata.layers["spliced"].toarray().T).T
    norm_factors = np.mean(anndata.obs["n_ucounts"]) / np.array(anndata.obs["n_ucounts"])
    anndata.layers["U_sz"] = (norm_factors * anndata.layers["unspliced"].toarray().T).T

def preprocess_for_phase_estimation(
    anndata,
    cycle_obj: Cycle,
    phase_obj: Phases,
    design_mtx,
    n_harmonics: int = 2,
    gene_selection_model: str = "all",
    normalize: bool = False,
    behavior: str = "intersection",
    noisemodel="NegativeBinomial",
    with_delta_nu: bool = True,
    condition_on={},
    μΔν=torch.tensor(0).float(),
    σΔν=torch.tensor(0.5).float(),
    gamma_alpha=torch.tensor(1.0).float(),
    gamma_beta=torch.tensor(2.0).float(),
    beta0=0.10, 
    beta1=0.90, 
    device=torch.device("cpu")
):
    μΔν = μΔν.to(device)
    σΔν = σΔν.to(device)
    gamma_alpha = gamma_alpha.to(device)
    gamma_beta = gamma_beta.to(device)

    if normalize:
        print("normalize")
        if ("S_sz" not in anndata.layers) or ("U_sz" not in anndata.layers):
            anndata.obs["n_scounts"] = anndata.layers["spliced"].toarray().sum(1)
            anndata.obs["n_ucounts"] = anndata.layers["unspliced"].toarray().sum(1)
            norm_factors = np.mean(anndata.obs["n_scounts"]) / np.array(anndata.obs["n_scounts"])
            anndata.layers["S_sz"] = (norm_factors * anndata.layers["spliced"].toarray().T).T
            norm_factors = np.mean(anndata.obs["n_ucounts"]) / np.array(anndata.obs["n_ucounts"])
            anndata.layers["U_sz"] = (norm_factors * anndata.layers["unspliced"].toarray().T).T

        S = torch.tensor(anndata.layers["S_sz"].astype(float)).to(device)
        U = torch.tensor(anndata.layers["U_sz"].astype(float)).to(device)
    else:
        try:
            S = torch.tensor(anndata.layers["spliced"].A.astype(np.int64)).to(device)
            U = torch.tensor(anndata.layers["unspliced"].A.astype(np.int64)).to(device)
        except:
            print("except")
            S = torch.tensor(anndata.layers["spliced"].astype(float)).to(device)
            U = torch.tensor(anndata.layers["unspliced"].astype(float)).to(device)

    S_UMI_per_cell = torch.tensor(anndata.layers["spliced"].sum(1).astype(np.int64)).T.float().to(device)
    U_UMI_per_cell = torch.tensor(anndata.layers["unspliced"].sum(1).astype(np.int64)).T.float().to(device)
    
    count_factor = torch.log(S_UMI_per_cell / torch.mean(S_UMI_per_cell)).to(device)
    count_factorU = torch.log(U_UMI_per_cell / torch.mean(U_UMI_per_cell)).to(device)
    
    anndata.layers["logS"] = np.log(S.cpu().numpy()+1+1e-16)
    anndata.layers["logU"] = np.log(U.cpu().numpy()+1+1e-16)
    
    if gene_selection_model=="all":
        model_fn = phase_latent_variable_model
        guide_fn = phase_latent_variable_guide
        cycle_objU = None
    elif gene_selection_model=="gmm":
        model_fn = phase_latent_variable_model_gmm
        guide_fn = phase_latent_variable_guide_gmm
    else:
        raise ValueError(f"{gene_selection_model=} is not a valid model")
    
    metapars = dict(
        Ng=len(cycle_obj),
        Nc=len(phase_obj),
        Nb=design_mtx.shape[-1],
        Db=design_mtx.T[:, None, :].float().to(device),
        cycle_prior=cycle_obj,
        phase_prior=phase_obj,
        μνg=cycle_obj.means_tensor.T[:, None, :].to(device),
        σνg=cycle_obj.stds_tensor.T[:, None, :].to(device),
        ϕxy_prior=phase_obj.phi_xy_tensor.T.to(device),
        gene_selection_model=gene_selection_model,
        model_fn=model_fn,
        guide_fn=guide_fn,
        num_harmonics_S=n_harmonics,
        basis_kind="fourier",
        noisemodel=noisemodel,
        gamma_alpha=gamma_alpha,
        gamma_beta=gamma_beta,
        device=device,
        kwargsζ=dict(num_harmonics=n_harmonics),
        σgc=torch.tensor(0.5).to(device),
        with_delta_nu=with_delta_nu,
        μΔν=μΔν.float().to(device),
        σΔν=σΔν.float().to(device),
        count_factor=count_factor[None, None, :],
        S=S.T.float().to(device),
        U=U.T.float().to(device),
        condition=np.array(list(condition_on.keys())),
        logS=torch.tensor(anndata.layers["logS"]).float().T.to(device),
        logU=torch.tensor(anndata.layers["logU"]).float().T.to(device),
        beta0=torch.tensor(beta0).to(device), 
        beta1=torch.tensor(beta1).to(device)
    )

    MetaparContainer = namedtuple("MetaparContainer", list(metapars.keys()))
    metapar_container = MetaparContainer(**metapars)

    return metapar_container

def preprocess_for_velocity_estimation(
    anndata,
    cycle_obj: Cycle,
    phase_obj: Phases,
    speed_obj: AngularSpeed,
    condition_design_mtx: np.ndarray,
    batch_design_mtx: np.ndarray,
    device=torch.device("cpu"),
    gene_selection_model: str = "all",
    null_cycle_obj: Cycle=None,
    n_harmonics: int = 2,
    norm_size: int = 1000,
    with_delta_nu: bool = True,
    count_factor: int=0,
    count_factorU: int=0,
    ω_n_harmonics: int=1,
    normalize: bool = False,
    behavior: str = "intersection",
    noisemodel="NegativeBinomial",
    condition_on={},
    μγ=torch.tensor(0.0).float(),
    σγ=torch.tensor(0.5).float(),
    μβ=torch.tensor(2.0).float(),
    σβ=torch.tensor(3.0).float(),
    μΔν=torch.tensor(0).float(),
    σΔν=torch.tensor(0.1).float(),
    gamma_alpha=torch.tensor(1.0).float(),
    gamma_beta=torch.tensor(2.0).float(),
    model_type: str = "lrmn",
    rho_mean = torch.tensor(4.0),
    rho_std = torch.tensor(1.0),
    rho_scale = torch.tensor(1.0),
    rho_rank = torch.tensor(5)
):
    cycle_obj, anndata = filter_shared_genes(cycle_obj, anndata, filter_type=behavior)
    
    if normalize:
        S = torch.tensor(anndata.layers["S_sz"].astype(np.int64), device=device)
        U = torch.tensor(anndata.layers["U_sz"].astype(np.int64), device=device)
    else:
        try:
            S = torch.tensor(anndata.layers["spliced"].A.astype(np.int64), device=device)
            U = torch.tensor(anndata.layers["unspliced"].A.astype(np.int64), device=device)
        except:
            S = torch.tensor(anndata.layers["spliced"].astype(np.int64), device=device)
            U = torch.tensor(anndata.layers["unspliced"].astype(np.int64), device=device)
        #count_factor = norm_size / S.sum(1)
    
    if model_type == "lrmn":
        model_fn = velocity_latent_variable_model_LRMN
        guide_fn = velocity_latent_variable_guide_LRMN
    elif gene_selection_model=="all":
        model_fn = velocity_latent_variable_model
        guide_fn = velocity_latent_variable_guide
    elif gene_selection_model=="gmm":
        model_fn = velocity_latent_variable_model_gmm
        guide_fn = velocity_latent_variable_guide_gmm
    else:
        raise ValueError(f"{gene_selection_model=} is not a valid model")
        
    anndata.layers["logS"] = np.log(S.cpu().numpy()+1+1e-16)
    anndata.layers["logU"] = np.log(U.cpu().numpy()+1+1e-16)
    ng = len(cycle_obj)
    metapars = dict(
        Ng = ng,
        Nc = len(phase_obj),
        Nhω = (ω_n_harmonics*2) + 1,
        Nb = batch_design_mtx.shape[-1],
        Nx = condition_design_mtx.shape[-1],
        D = condition_design_mtx.T[:, None, None, :].clone().detach().to(device),
        Db = batch_design_mtx.T[:, None, None, None, :].clone().detach().to(device),
        ν = cycle_obj.means_tensor.T.unsqueeze(-2).to(device),
        cycle_prior=cycle_obj,
        phase_prior=phase_obj,
        speed_prior=speed_obj,
        gene_selection_model=gene_selection_model,
        model_fn=model_fn,
        guide_fn=guide_fn,
        with_delta_nu=with_delta_nu,
        μΔν=μΔν.to(device), 
        σΔν=σΔν.to(device),        
        μγ=μγ.detach().clone().float().repeat([ng, 1]).to(device),
        σγ=σγ.detach().clone().float().repeat([ng, 1]).to(device),
        μβ=μβ.detach().clone().float().repeat([ng, 1]).to(device),
        σβ=σβ.detach().clone().float().repeat([ng, 1]).to(device),
        μνω=speed_obj.means_tensor.T.unsqueeze(-1).unsqueeze(-1).to(device),
        σνω=speed_obj.stds_tensor.T.unsqueeze(-1).unsqueeze(-1).to(device),
        μνg=cycle_obj.means_tensor.T[:, None, :].to(device),
        σνg=cycle_obj.stds_tensor.T[:, None, :].to(device),
        ϕxy_prior=phase_obj.phi_xy_tensor.T.to(device),
        basis_kind="fourier",
        num_harmonics=n_harmonics,
        noisemodel=noisemodel,
        gamma_alpha=gamma_alpha.to(device),
        gamma_beta=gamma_beta.to(device),
        count_factor=count_factor.clone().detach().to(device),
        kwargsζ=dict(num_harmonics=n_harmonics),
        kwargsζ_dϕ = dict(num_harmonics=n_harmonics),
        kwargsζω = dict(num_harmonics=ω_n_harmonics),
        σₛgc=torch.tensor(0.1, device=device),
        σᵤgc=torch.tensor(0.1, device=device),
        S=S.T.float().to(device),
        U=U.T.float().to(device),
        logS=torch.tensor(anndata.layers["logS"]).float().T.to(device),
        logU=torch.tensor(anndata.layers["logU"]).float().T.to(device),
        condition=np.array(list(condition_on.keys())),
        device=device,
        model_type=model_type,
        rho_mean = rho_mean.to(device),
        rho_std = rho_std.to(device),
        rho_scale = rho_scale.to(device),
        rho_rank = rho_rank.to(device)
    )

    MetaparContainer = namedtuple("MetaparContainer", list(metapars.keys()))
    metapar_container = MetaparContainer(**metapars)
    return metapar_container
