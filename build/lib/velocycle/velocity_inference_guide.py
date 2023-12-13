#!/usr/bin/env python
# coding: utf-8

import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints

def velocity_latent_variable_guide(mp):
    """
    Defines the velocity latent variable guide for use with Pyro.

    Parameters:
    - mp: A structure containing metaparameters for the guide.
    """
    # Plates initialization
    device=mp.device
   
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device)
    harmonics_plate = pyro.plate("harmonics", mp.Nhω, dim=-3, device=device)
    conditions_plate = pyro.plate("conditions", mp.Nx, dim=-4, device=device)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-5, device=device)

    logγg_locs = pyro.param("logγg_locs", mp.μγ.detach().clone().to(device))
    logβg_locs = pyro.param("logβg_locs", mp.μβ.detach().clone().to(device))

    logγg_scales = pyro.param("logγg_scales", mp.σγ.detach().clone().to(device), constraint=constraints.positive)
    logβg_scales = pyro.param("logβg_scales", mp.σβ.detach().clone().to(device), constraint=constraints.positive)

    ν_locs = pyro.param("ν_locs", mp.μνg.detach().clone().to(device))
    ν_scales = pyro.param("ν_scales", mp.σνg.detach().clone().to(device), constraint=constraints.positive)

    if mp.with_delta_nu:
        Δν_locs = pyro.param("Δν_locs", torch.ones((mp.Nb, 1, 1, mp.Ng, 1), device=device) * mp.μΔν.to(device))
    
    ϕxy_locs = pyro.param("ϕxy_locs", mp.φxy_prior.detach().clone().to(device))

    νω_locs = pyro.param("νω_locs", mp.μνω.detach().clone().to(device))
    νω_scales = pyro.param("νω_scales", mp.σνω.detach().clone().to(device), constraint=constraints.positive)

    if mp.noisemodel == "NegativeBinomial":
        shape_inv_locs = pyro.param("shape_inv_locs", (torch.ones((mp.Ng, 1), device=device) * mp.gamma_alpha/mp.gamma_beta).to(device), constraint=constraints.positive)

    with gene_plate:
        logγg = pyro.sample("logγg", dist.Normal(logγg_locs, logγg_scales))
        logβg = pyro.sample("logβg", dist.Normal(logβg_locs, logβg_scales))
        
        ν = pyro.sample("ν", dist.Normal(ν_locs, ν_scales).to_event(1))

        if mp.with_delta_nu:
            with batches_plate:
                Δν = pyro.sample("Δν", dist.Delta(Δν_locs))

        if mp.noisemodel == "NegativeBinomial":
            shape_inv = pyro.sample("shape_inv", dist.Delta(shape_inv_locs))

    with harmonics_plate:
        with conditions_plate:
            νω = pyro.sample("νω", dist.Normal(νω_locs, νω_scales))

    with cell_plate:
        ϕxy = pyro.sample("ϕxy", dist.Normal(ϕxy_locs, torch.tensor(1.0).to(device)).to_event(1))    
        
def velocity_latent_variable_guide_LRMN(mp):
    """
    Defines the low-rank multivariate normal (LRMN) variant of the velocity latent variable guide.
    """
    
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2)
    harmonics_plate = pyro.plate("harmonics", mp.Nhω, dim=-3)
    conditions_plate = pyro.plate("conditions", mp.Nx, dim=-4)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-5)

    device = mp.device

    ν_locs = pyro.param("ν_locs", mp.μνg.detach().clone())
    ν_scales = pyro.param("ν_scales", mp.σνg.detach().clone(), constraint=constraints.positive)
    
    if mp.with_delta_nu:
        Δν_locs = pyro.param("Δν_locs", torch.ones((mp.Nb, 1, 1, mp.Ng, 1), device=device) * mp.μΔν.to(device))
    
    ϕxy_locs = pyro.param("ϕxy_locs", mp.φxy_prior.detach().clone())
    
    logβg_locs = pyro.param("logβg_locs", mp.μβ.detach().clone())
    logβg_scales = pyro.param("logβg_scales", mp.σβ.detach().clone(), constraint=constraints.positive)
    
    lrmv_dims = mp.Ng + (mp.Nhω*mp.Nx)
    loc = pyro.param("loc", torch.hstack([mp.μγ.squeeze().detach().clone(), mp.μνω.squeeze().detach().clone().flatten()]))
    cov_factor = pyro.param("cov_factor", torch.clip(torch.normal(torch.zeros((lrmv_dims, mp.rho_rank), device=device), 
                                                                  torch.ones((lrmv_dims, mp.rho_rank), device=device)*0.02), min=0, max=None,).to(device), constraint=constraints.positive)
    cov_diag = pyro.param("cov_diag", (torch.hstack([mp.σγ.squeeze().detach().clone(), mp.σνω.squeeze().detach().clone().flatten()])**2).to(device), constraint=constraints.positive)
    
    LRMV_X = dist.LowRankMultivariateNormal(loc=loc,
                                   cov_factor = cov_factor,
                                   cov_diag = cov_diag).rsample()

    rho_real_loc = pyro.param("rho_real_loc", torch.ones(mp.Ng, device=device)*mp.rho_mean) # _loc

    if mp.noisemodel == "NegativeBinomial":
        shape_inv_locs = pyro.param("shape_inv_locs", torch.ones((mp.Ng, 1), device=device) * mp.gamma_alpha/mp.gamma_beta, constraint=constraints.positive)
    
    with gene_plate: 
        logγg = pyro.sample("logγg", dist.Delta(LRMV_X[:mp.Ng].unsqueeze(-1)))
        ν = pyro.sample("ν", dist.Normal(ν_locs, ν_scales).to_event(1))
        
        rho_real = pyro.sample("rho_real", dist.Delta(rho_real_loc.unsqueeze(-1)))
        rho = torch.sigmoid(rho_real/mp.rho_scale) * 1.998 - 0.999
        
        if mp.with_delta_nu:
            with batches_plate:
                Δν = pyro.sample("Δν", dist.Delta(Δν_locs.to(device)))
            
        if mp.noisemodel == "NegativeBinomial":
            shape_inv = pyro.sample("shape_inv", dist.Delta(shape_inv_locs))
    
    loc_gammas = loc[:mp.Ng]
    scale_gammas_mtx = cov_factor @ cov_factor.T + torch.diag(cov_diag)
    scale_gammas = torch.sqrt(torch.diag(scale_gammas_mtx)[:mp.Ng])
    
    mu_beta_given_gamma = logβg_locs.squeeze() + rho.squeeze() * logβg_scales.squeeze() * (logγg.squeeze() - loc_gammas) / scale_gammas
    std_beta_given_gamma = logβg_scales.squeeze() * torch.sqrt(1 - rho.squeeze()**2)
    
    with gene_plate:
        logβg = pyro.sample("logβg", dist.Normal(mu_beta_given_gamma.unsqueeze(-1), std_beta_given_gamma.unsqueeze(-1)))
    
    with harmonics_plate:
        with conditions_plate:
            if mp.Nhω > 1:
                if mp.Nx > 1: # Multiple conditions
                    νω = pyro.sample("νω", dist.Delta(LRMV_X[mp.Ng:].reshape((mp.Nx, mp.Nhω)).unsqueeze(-1).unsqueeze(-1)))
                else:
                    νω = pyro.sample("νω", dist.Delta(LRMV_X[mp.Ng:].unsqueeze(-1).unsqueeze(-1)))
            else:
                if mp.Nx > 1:
                    νω = pyro.sample("νω", dist.Delta(LRMV_X[mp.Ng:].reshape((mp.Nx, mp.Nhω)).unsqueeze(-1).unsqueeze(-1)))
                else:
                    νω = pyro.sample("νω", dist.Delta(LRMV_X[mp.Ng:].unsqueeze(-1).unsqueeze(-1)))
    with cell_plate:
        ϕxy = pyro.sample("ϕxy", dist.Normal(ϕxy_locs, 1.0).to_event(1))