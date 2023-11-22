#!/usr/bin/env python
# coding: utf-8

import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.infer import config_enumerate

def phase_latent_variable_guide(mp):
    """
    Define a guide for a phase latent variable model in Pyro.

    This function specifies the variational distribution (guide) for the latent variables in a 
    phase latent variable model. It initializes plates for cells, genes, and batches, and defines 
    Pyro parameters and samples for various latent variables like ν, a, Δν, ϕxy, and shape_inv 
    depending on the model properties.

    Parameters:
    - mp (object): A model parameter object containing attributes such as device, Nc (number of cells),
                   Ng (number of genes), Nb (number of batches), num_harmonics_S, μνg, σνg, μa, μΔν,
                   φxy_prior, and parameters related to the noise model.

    Returns:
    - None: This function defines Pyro samples and does not return any value.
    """
    
    device = mp.device  # Assuming mp object has an attribute specifying the device
    # Plates initialization
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-3, device=device)

    Nharm = mp.num_harmonics_S * 2 + 1

    ν_locs = pyro.param("ν_locs", mp.μνg.detach().clone().to(device))
    ν_scales = pyro.param("ν_scales", mp.σνg.detach().clone().to(device), constraint=constraints.positive)

    if mp.with_delta_nu:
            with batches_plate:
                Δν_locs = pyro.param("Δν_locs", torch.ones((mp.Nb, mp.Ng, 1), device=device) * mp.μΔν)
    ϕxy_locs = pyro.param("ϕxy_locs", mp.φxy_prior.detach().clone().to(device))

    if mp.noisemodel == "NegativeBinomial":
        shape_inv_locs = pyro.param("shape_inv_locs", torch.ones((mp.Ng, 1), device=device) * mp.gamma_alpha/mp.gamma_beta, constraint=constraints.positive)

    with gene_plate:
        ν = pyro.sample("ν", dist.Normal(ν_locs, ν_scales).to_event(1))
        if mp.noisemodel == "NegativeBinomial":
            shape_inv = pyro.sample("shape_inv", dist.Delta(shape_inv_locs))
        if mp.with_delta_nu:
            with batches_plate:
                Δν = pyro.sample("Δν", dist.Delta(Δν_locs))

    with cell_plate:
        ϕxy = pyro.sample("ϕxy", dist.Normal(ϕxy_locs, torch.tensor(1.0, device=device)).to_event(1))  # SHOULD THIS BE A DELTA???
        
def clipped_sigmoid(x):
    """
    Apply a sigmoid function to input tensor 'x' with clipping to avoid numerical instability.

    This function computes a sigmoid transformation of the input tensor 'x', while clipping the 
    input to avoid extreme values that could lead to numerical instability. The output is also 
    clipped to ensure it remains within the range [eps, 1 - eps], where eps is a small positive 
    number determined by the floating point information of the input tensor's data type.

    Parameters:
    - x (torch.Tensor): Input tensor to which the sigmoid function is applied.

    Returns:
    - torch.Tensor: The result of applying the clipped sigmoid function to 'x'.
    """
    
    finfo = torch.finfo(x.dtype)
    y = x / 1.
    z = torch.clamp(y, min=finfo.min+10*finfo.eps, max=finfo.max-10*finfo.eps)
    return torch.clamp(torch.sigmoid(z), min=finfo.eps, max=1.-finfo.eps)

@config_enumerate
def phase_latent_variable_guide_lba(mp):
    """
    Define a guide for a Latent Bernoulli Allocation (LBA) phase latent variable in Pyro.

    Similar to 'phase_latent_variable_guide', this function specifies the variational distribution
    for a LBA phase latent variable model.
    
    Parameters:
    - mp (object): A model parameter object similar to the one used in 'phase_latent_variable_guide'.

    Returns:
    - None: This function defines Pyro samples for a LBA phase latent variable model and does not 
            return any value.
    """
    
    device = mp.device  # Assuming mp object has an attribute specifying the device
    # Plates initialization
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-3, device=device)

    Nharm = mp.num_harmonics_S * 2 + 1

    ν_locs = pyro.param("ν_locs", mp.μνg.detach().clone().to(device))
    ν_scales = pyro.param("ν_scales", mp.σνg.detach().clone().to(device), constraint=constraints.positive)

    if mp.with_delta_nu:
            with batches_plate:
                Δν_locs = pyro.param("Δν_locs", torch.ones((mp.Nb, mp.Ng, 1), device=device) * mp.μΔν)
    ϕxy_locs = pyro.param("ϕxy_locs", mp.φxy_prior.detach().clone().to(device))

    if mp.noisemodel == "NegativeBinomial":
        shape_inv_locs = pyro.param("shape_inv_locs", torch.ones((mp.Ng, 1), device=device) * mp.gamma_alpha/mp.gamma_beta, constraint=constraints.positive)

    
    avg_p =  torch.tensor(mp.beta0 / (mp.beta0+mp.beta1), device=device)
    logit_avg = (torch.log(avg_p / (1-avg_p))).to(device)
    logit_locs = pyro.param("logit_locs", torch.zeros((mp.Ng, 1), device=device))
    
    with gene_plate:
        ν = pyro.sample("ν", dist.Normal(ν_locs, ν_scales).to_event(1))
        if mp.noisemodel == "NegativeBinomial":
            shape_inv = pyro.sample("shape_inv", dist.Delta(shape_inv_locs))
        if mp.with_delta_nu:
            with batches_plate:
                Δν = pyro.sample("Δν", dist.Delta(Δν_locs))
               
        periodic_prob = pyro.sample('prob', dist.Delta(clipped_sigmoid(logit_locs+(logit_avg*1.)).to(device)))
        
    with cell_plate:
        ϕxy = pyro.sample("ϕxy", dist.Normal(ϕxy_locs, torch.tensor(1.0, device=device)).to_event(1))