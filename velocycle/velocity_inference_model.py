#!/usr/bin/env python
# coding: utf-8

# Standard imports
import numpy as np
import torch
import logging
import pyro
import copy
import collections

# Pyro imports
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide import init_to_mean, init_to_median
import pyro.distributions as dist

# VeloCycle imports
from .plots import live_plot
from .cycle import Cycle
from .phases import Phases
from .angularspeed import AngularSpeed
from .velocity_inference_guide import velocity_latent_variable_guide 
from .utils import (
    pack_direction,
    torch_fourier_basis,
    torch_basis
)


class VelocityFitModel:
    """
    A class for executing velocity-learning of the cell cycle angular speed and velocity kinetic parameters.

    This class encapsulates the model, guide, and fitting process for velocity data. It supports conditioning on
    certain parameters, early exit from the fitting process based on convergence criteria, and extraction of
    posterior distributions.

    Attributes:
    - model (function): The Pyro model function.
    - guide (function): The Pyro guide function (variational distribution).
    - posterior (dict or None): Stores the posterior distribution after fitting.
    - condition (dict): The data to condition the model on.
    - condition_on (list): Keys of the data to condition the model on.
    - metaparams: Metaparameters for the model.
    - early_exit (bool): Whether to enable early exit from fitting based on convergence.
    - get_posterior (bool): Whether to compute the posterior after fitting.
    - num_samples (int): Number of samples to draw for posterior computations.
    - n_per_bin (int): Number of samples per bin for posterior computations.

    Parameters:
    - metaparams: Metaparameters for initializing the model.
    - condition_on (dict, optional): Data to condition the model on. Defaults to an empty dictionary.
    - early_exit (bool, optional): Flag to enable early exit. Defaults to True.
    - get_posterior (bool, optional): Flag to compute posterior after fitting. Defaults to True.
    - num_samples (int, optional): Number of samples for posterior computation. Defaults to 500.
    - n_per_bin (int, optional): Number of samples per bin for posterior computations. Defaults to 50.
    """
    def __init__(self, metaparams, condition_on={}, early_exit=False, get_posterior=True, num_samples=500, n_per_bin=50):
        if len(condition_on)==0:
            self.model = metaparams.model_fn
            self.guide = metaparams.guide_fn
        else:
            self.model = poutine.condition(metaparams.model_fn, data=condition_on)
            self.guide = poutine.block(metaparams.guide_fn, hide=list(condition_on.keys()))
        self.posterior = None
        self.condition = condition_on
        self.condition_on = list(condition_on.keys())
        self.metaparams = metaparams
        self.early_exit = early_exit
        self.get_posterior = get_posterior
        self.num_samples = num_samples
        self.n_per_bin = n_per_bin

    def fit(
        self,
        optimizer,
        loss=pyro.infer.Trace_ELBO(num_particles = 1), # change the num_particles?? change only halfway through fitting from 3 to 1 using a new svi object with a different loss
        num_steps=1000,
        intermediate_output_step_size=500,
        store_output=False,
        verbose=True,
    ):
        """
        Fits the model to the data using Stochastic Variational Inference (SVI).

        Parameters:
        - optimizer: The optimizer to use for SVI.
        - loss (optional): The loss function for SVI. Defaults to Trace_ELBO with one particle.
        - num_steps (int, optional): Number of steps to run SVI. Defaults to 1000.
        - intermediate_output_step_size (int, optional): Steps interval for intermediate output. Defaults to 500.
        - store_output (bool, optional): Flag to store intermediate outputs. Defaults to False.
        - verbose (bool, optional): Enables verbose output. Defaults to True.

        This function performs SVI on the model and guide, using the provided optimizer and loss function. It supports 
        early exit based on convergence criteria, storing intermediate outputs, and verbose logging. Post-fit, it updates 
        the model's attributes with results and posterior distributions.
        """
        # We call the guide here so that it's in the same namespace
        if self.guide is None:
            self.guide = AutoNormal(
                self.model, init_loc_fn=init_to_mean(fallback=init_to_median(num_samples=50))
            )

        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss)

        losses = []
        intermediate_output = []
        plotting_data = collections.defaultdict(list)
        early_exit_bool=False
        model_type = self.metaparams.model_type
        for step in range(num_steps):
            # loss = svi.loss(self.conditioned_model, self.guide, self.metaparams, init_to_mean)
            loss = svi.step(self.metaparams)
            losses.append(loss)
            if store_output:
                if step % intermediate_output_step_size == 0:
                    logging.info("Elbo loss: {}".format(loss))

                    if model_type=="lrmn":
                        if self.metaparams.with_delta_nu:
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'Δν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv', 'rho_real'])
                        else: 
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv', 'rho_real'])
                    else:
                        if self.metaparams.with_delta_nu:
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'Δν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv'])
                        else:
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv'])
                    velo_pps_cpu = {k: v.cpu() for k, v in velo_pps.items()}  # Move samples to CPU
                    intermediate_output.append(velo_pps_cpu)
                    
            if verbose:
                if step > 5 and step % 40 == 0:
                    plotting_data["ELBO"] = losses
                    live_plot(plotting_data)
            if early_exit_bool:
                if np.abs(np.mean(losses[-100:])-np.mean(losses[-10:]))<5:
                    break
            elif step > 200 and self.early_exit:
                early_exit_bool=True

        self.losses = losses
        
        # Put estimations in new objects
        self.phis_pyro = pyro.param("ϕxy_locs").detach().squeeze().cpu().numpy().T
        self.fourier_coef = pyro.param("ν_locs").detach().squeeze().cpu().numpy().T
        self.fourier_coef_sd = pyro.param("ν_scales").detach().squeeze().cpu().numpy().T
    
        new_cycle = Cycle.from_array(self.fourier_coef, self.fourier_coef_sd, self.metaparams.cycle_prior.genes)
        new_phase = Phases.from_array(self.phis_pyro, cell_names=self.metaparams.phase_prior.phi_xy.columns)
            
        self.disp_pyro = pyro.param("shape_inv_locs").detach().squeeze().cpu().numpy().T
        if self.metaparams.with_delta_nu:
            self.delta_nus = pyro.param("Δν_locs").detach().unsqueeze(-3).unsqueeze(-4).float().cpu().numpy()

        model_type = self.metaparams.model_type
        if model_type != "lrmn":
            self.log_gammas = pyro.param("logγg_locs").detach().squeeze().cpu().numpy().T
            new_cycle.set_log_gammas(self.log_gammas)

            self.velocity_coef = pyro.param("νω_locs").detach().unsqueeze(-3).unsqueeze(-4).float().cpu().numpy()
            self.velocity_coef_sd = pyro.param("νω_scales").detach().unsqueeze(-3).unsqueeze(-4).float().cpu().numpy()
           
            new_speed = AngularSpeed.from_array(condition_names=self.metaparams.speed_prior.conditions,
                                            means_array=self.velocity_coef.squeeze(),
                                            stds_array=self.velocity_coef_sd.squeeze(),
                                            Nhω=self.metaparams.Nhω)
            self.speed_pyro = new_speed
        
        
        self.log_betas = pyro.param("logβg_locs").detach().squeeze().cpu().numpy().T 
        new_cycle.set_log_betas(self.log_betas)
        new_cycle.set_disp_pyro(self.disp_pyro)

        self.cycle_pyro = new_cycle
        self.phase_pyro = new_phase
        
        if self.get_posterior:
            if False: # cpu
                self.posterior = self.sample_posterior(num_samples=self.num_samples)
    
                self.metaparams_avg = copy.deepcopy(self.metaparams)
                self.metaparams_avg.count_factor[:] = torch.mean(self.metaparams_avg.count_factor)
    
                self.posterior_avg = self.sample_posterior(num_samples=self.num_samples, mp=self.metaparams_avg)
            
                if (self.metaparams.gene_selection_model=="lba"):
                    self.periodic = self.posterior["periodic"].mean(0).squeeze().cpu().numpy()
                    self.periodic_probs = pyro.param("logit_locs").detach().squeeze().cpu().numpy().T
            else: # gpu
                nsamples = self.num_samples
                n_per_bin = self.n_per_bin # 50 posterior samples per call
                nbins = int(np.ceil(nsamples/n_per_bin))
                
                self.metaparams_avg = copy.deepcopy(self.metaparams)
                self.metaparams_avg.count_factor[:] = torch.mean(self.metaparams_avg.count_factor)
                
                velo_pps_cpu_dict_list = []
                for curr_bin in range(nbins):
                    if model_type=="lrmn":
                        if self.metaparams.with_delta_nu:
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'Δν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv', 'rho_real'])
                        else: 
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv', 'rho_real'])
                    else:
                        if self.metaparams.with_delta_nu:
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'Δν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv'])
                        else:
                            velo_pps = self.sample_posterior(num_samples=n_per_bin, rs=['logγg', 'logβg', 'νω', 'γg', 'ν', 'ϕxy', 'ϕ', 
                                                                                'ζ', 'ζ_dϕ', 'ζω', 'ω', 'shape_inv'])
                    velo_pps_cpu = {k: v.cpu() for k, v in velo_pps.items()}  # Move samples to CPU
                
                    del velo_pps
                    torch.cuda.empty_cache()
                    velo_pps_cpu_dict_list.append(velo_pps_cpu)

                velo_pps_cpu_full = {}
                for k in velo_pps_cpu_dict_list[0].keys():
                    concat_pps_tensor = torch.vstack([velo_pps_cpu_dict_list[i][k] for i in range(nbins)])
                    velo_pps_cpu_full[k] = concat_pps_tensor
                
                ν = pyro.param("ν_locs").detach().cpu()
                if self.metaparams.with_delta_nu:
                    Δν = pyro.param("Δν_locs").detach().cpu()
                
                ζ = torch_basis(self.phase_pyro.phis, der=0, kind=self.metaparams.basis_kind, device=torch.device("cpu"), **self.metaparams.kwargsζ)
                
                if self.metaparams.with_delta_nu:
                    ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + torch.einsum("bxhgc,bxhgc->gc", self.metaparams.Db.to(torch.device("cpu")), Δν) + self.metaparams.count_factor.to(torch.device("cpu"))
                    ElogS2 = torch.einsum("...gch,ch->gc", ν, ζ) + torch.einsum("bxhgc,bxhgc->gc", self.metaparams.Db.to(torch.device("cpu")), Δν) + self.metaparams_avg.count_factor.to(torch.device("cpu"))
                else:
                    ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + self.metaparams.count_factor.to(torch.device("cpu"))
                    ElogS2 = torch.einsum("...gch,ch->gc", ν, ζ) + self.metaparams_avg.count_factor.to(torch.device("cpu"))
                
                ζ_dϕ = torch_basis(self.phase_pyro.phis, der=1, kind=self.metaparams.basis_kind, device=torch.device("cpu"), **self.metaparams.kwargsζ_dϕ)
                γg = velo_pps_cpu_full["γg"].mean(0).squeeze().cpu().unsqueeze(-1)
                logβg = velo_pps_cpu_full["logβg"].mean(0).squeeze().cpu().unsqueeze(-1)
                ζω = torch_basis(self.phase_pyro.phis, der=0, kind=self.metaparams.basis_kind, device=torch.device("cpu"), **self.metaparams.kwargsζω).T
                νω = velo_pps_cpu_full["νω"].mean(0)
                ω = torch.einsum("...xhgc,hc,xhgc->gc", [νω, ζω, self.metaparams.D.cpu()])
                ElogU = -logβg + torch.log(torch.relu(torch.einsum("gch,ch->gc", ν, ζ_dϕ) * ω + γg) + 1e-5) + ElogS
                ElogU2 = -logβg + torch.log(torch.relu(torch.einsum("gch,ch->gc", ν, ζ_dϕ) * ω + γg) + 1e-5) + ElogS2
                velo_pps_cpu_full["ElogS"] = ElogS.squeeze()
                velo_pps_cpu_full["ElogU"] = ElogU.squeeze()
                velo_pps_cpu_full["ElogS2"] = ElogS2.squeeze()
                velo_pps_cpu_full["ElogU2"] = ElogU2.squeeze()

                self.posterior = velo_pps_cpu_full

            if model_type == "lrmn":
                self.log_gammas = self.posterior["logγg"].mean(0).squeeze().detach().numpy().T
                new_cycle.set_log_gammas(self.log_gammas)

                self.velocity_coef = self.posterior["νω"].mean(0).float().cpu().numpy() 
               
                new_speed = AngularSpeed.from_array(condition_names=self.metaparams.speed_prior.conditions,
                                                means_array=self.velocity_coef.squeeze(),
                                                stds_array=self.posterior["νω"].std(0).float().cpu().squeeze().numpy(),
                                                Nhω=self.metaparams.Nhω)
                self.speed_pyro = new_speed
            
        if store_output:
                return intermediate_output
            
    def sample_posterior(self, num_samples=1, rs=None, mp=None, take_mean=True):
        if mp is None: 
            mp = self.metaparams
        if rs is None:
            full_predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        else:
            full_predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=rs)
        full_pps = full_predictive(mp)
        
        for k in full_pps.keys():
            full_pps[k] = full_pps[k].cpu()
        
        return full_pps
    
    def _check_model(self, m, *args):
        pyro.clear_param_store()
        trace = poutine.trace(m).get_trace(*args)
        print(trace.format_shapes())

    def check_model(self):
        return self._check_model(self.model, self.metaparams)

    def check_guide(self):
        return self._check_model(self.guide, self.metaparams)

def velocity_latent_variable_model(mp, init_loc_fn=init_to_mean(fallback=init_to_median(num_samples=50))):
    """
    Defines the velocity latent variable model for use with Pyro.

    Parameters:
    - mp: A structure containing metaparameters for the model.
    - init_loc_fn (optional): A function for initializing location parameters. Defaults to mean initialization with median fallback.
    """
    device=mp.device
    
    # Plates initialization
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device)
    harmonics_plate = pyro.plate("harmonics", mp.Nhω, dim=-3, device=device)
    conditions_plate = pyro.plate("conditions", mp.Nx, dim=-4, device=device)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-5, device=device)
    
    # Generate beta and gamma distributions
    with gene_plate:
        logγg = pyro.sample("logγg", dist.Normal(mp.μγ.to(device), mp.σγ.to(device)))
        logβg = pyro.sample("logβg", dist.Normal(mp.μβ.to(device), mp.σβ.to(device)))
            
        γg = torch.exp(logγg)
        pyro.deterministic("γg", γg)
        ν = pyro.sample("ν", dist.Normal(mp.μνg.to(device), mp.σνg.to(device)).to_event(1))
        
        if mp.with_delta_nu:
            with batches_plate:
                Δν = pyro.sample("Δν", dist.Normal(torch.tensor(0., device=device), torch.tensor(0.01, device=device)))
    
    # Build gene harmonics
    if mp.basis_kind == "fourier":
        with cell_plate:
            ϕxy = pyro.sample("ϕxy", dist.Normal(mp.ϕxy_prior.to(device), torch.tensor(1.0, device=device)).to_event(1))
        ϕ = pack_direction(ϕxy)
        pyro.deterministic("ϕ", ϕ)
    else:
        with cell_plate:
            ϕ = pyro.sample("ϕ", dist.Uniform(0, 2*mp.pi))
    
    ζ = torch_basis(ϕ, der=0, kind=mp.basis_kind, device=device, **mp.kwargsζ)
    ζ_dϕ = torch_basis(ϕ, der=1, kind=mp.basis_kind, device=device, **mp.kwargsζ_dϕ)

    pyro.deterministic("ζ", ζ)
    pyro.deterministic("ζ_dϕ", ζ_dϕ)

    # Build velocity harmonics
    with harmonics_plate:
        with conditions_plate:
            νω = pyro.sample("νω", dist.Normal(mp.μνω.to(device), mp.σνω.to(device)))
    
    ζω = torch_basis(ϕ, der=0, kind=mp.basis_kind, device=device, **mp.kwargsζω).T
    
    pyro.deterministic("ζω", ζω)

    if mp.with_delta_nu:
        ElogS = torch.einsum("...gch,...ch->gc", ν, ζ) + torch.einsum("bxhgc,bxhgc->gc", mp.Db, Δν) + mp.count_factor
    else:
        ElogS = torch.einsum("...gch,...ch->gc", ν, ζ) + mp.count_factor
    pyro.deterministic("ElogS", ElogS)
    
    ω = torch.einsum("...xhgc,hc...,xhgc->gc", [νω, ζω, mp.D])
    pyro.deterministic("ω", ω)
    
    ElogU = -logβg + torch.log(torch.relu(torch.einsum("...gch,...ch->gc", ν, ζ_dϕ) * ω + γg) + 1e-5) + ElogS
    pyro.deterministic("ElogU", ElogU)

    # Add noise to the expectation
    if mp.noisemodel == "Lognormal":
        with gene_plate, cell_plate:
            logS = pyro.sample("logS", dist.Normal(ElogS, mp.σₛgc.to(device)), obs=mp.logS.to(device))
            logU = pyro.sample("logU", dist.Normal(ElogU, mp.σᵤgc.to(device)), obs=mp.logU.to(device))
    elif mp.noisemodel == "Poisson":
        with gene_plate, cell_plate:
            pyro.sample("S", dist.Poisson(torch.exp(ElogS)), obs=mp.S.to(device))
            pyro.sample("U", dist.Poisson(torch.exp(ElogU)), obs=mp.U.to(device))
    elif mp.noisemodel == "NegativeBinomial":
        with gene_plate:
            # the logic is that 1/shape is the dispersion parameter which 'should' be less than 1
            shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        with cell_plate, gene_plate:
            pyro.sample("S", dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogS))), obs=mp.S.to(device))
            pyro.sample("U", dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogU))), obs=mp.U.to(device))
    else:
        raise ValueError(f"{mp.noisemodel} not allowed")

def velocity_latent_variable_model_LRMN(mp, init_loc_fn=init_to_mean(fallback=init_to_median(num_samples=50))):
    """
    Defines the low-rank multivariate normal (LRMN) variant of the velocity latent variable model.
    """
    
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2)
    harmonics_plate = pyro.plate("harmonics", mp.Nhω, dim=-3)
    conditions_plate = pyro.plate("conditions", mp.Nx, dim=-4)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-5)

    device = mp.device
    
    # Generate beta and gamma distributions
    with gene_plate:
        logγg = pyro.sample("logγg", dist.Normal(mp.μγ.to(device), mp.σγ.to(device)))
        logβg = pyro.sample("logβg", dist.Normal(mp.μβ.to(device), mp.σβ.to(device)))
        
        rho_real = pyro.sample("rho_real", dist.Normal(mp.rho_mean, mp.rho_std))
        
        γg = torch.exp(logγg)
        pyro.deterministic("γg", γg)
        ν = pyro.sample("ν", dist.Normal(mp.μνg.to(device), mp.σνg.to(device)).to_event(1))
        
        if mp.with_delta_nu:
            with batches_plate:
                Δν = pyro.sample("Δν", dist.Normal(torch.tensor(0.0, device=device), torch.tensor(0.01, device=device)))
    
    # Build gene harmonics
    if mp.basis_kind == "fourier":
        with cell_plate:
            ϕxy = pyro.sample("ϕxy", dist.Normal(mp.ϕxy_prior, 1.0).to_event(1))
        ϕ = pack_direction(ϕxy)
        pyro.deterministic("ϕ", ϕ)
    else:
        with cell_plate:
            ϕ = pyro.sample("ϕ", dist.Uniform(0, 2*mp.pi))

    ζ = torch_basis(ϕ, der=0, kind=mp.basis_kind, device=device, **mp.kwargsζ)
    ζ_dϕ = torch_basis(ϕ, der=1, kind=mp.basis_kind, device=device, **mp.kwargsζ_dϕ)

    pyro.deterministic("ζ", ζ)
    pyro.deterministic("ζ_dϕ", ζ_dϕ)
    
    # Build velocity harmonics
    with harmonics_plate:
        with conditions_plate:
            νω = pyro.sample("νω", dist.Normal(mp.μνω.to(device), mp.σνω.to(device)))
    
    ζω = torch_basis(ϕ, der=0, kind=mp.basis_kind, device=device, **mp.kwargsζω).T
    
    pyro.deterministic("ζω", ζω)
    
    if mp.with_delta_nu:
        ElogS = torch.einsum("...gch,...ch->gc", ν, ζ) + torch.einsum("bxhgc,bxhgc->gc", mp.Db, Δν) + mp.count_factor
    else:
        ElogS = torch.einsum("...gch,...ch->gc", ν, ζ) + mp.count_factor
    pyro.deterministic("ElogS", ElogS)
    
    ω = torch.einsum("...xhgc,hc...,xhgc->gc", [νω, ζω, mp.D])
    pyro.deterministic("ω", ω)
    ElogU = -logβg + torch.log(torch.relu(torch.einsum("...gch,...ch->gc", ν, ζ_dϕ) * ω + γg) + 1e-5) + ElogS
    pyro.deterministic("ElogU", ElogU)
    
    # Add noise to the expectation
    if mp.noisemodel == "Lognormal":
        with gene_plate, cell_plate:
            logS = pyro.sample("logS", dist.Normal(ElogS, mp.σₛgc), obs=mp.logS)
            logU = pyro.sample("logU", dist.Normal(ElogU, mp.σᵤgc), obs=mp.logU)
    elif mp.noisemodel == "Poisson":
        with gene_plate, cell_plate:
            pyro.sample("S", dist.Poisson(torch.exp(ElogS)), obs=mp.S)
            pyro.sample("U", dist.Poisson(torch.exp(ElogU)), obs=mp.U)
    elif mp.noisemodel == "NegativeBinomial":
        with gene_plate:
            # the logic is that 1/shape is the dispersion parameter which 'should' be less than 1
            shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha, mp.gamma_beta))
        with cell_plate, gene_plate:
            pyro.sample("S", dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogS))), obs=mp.S)
            pyro.sample("U", dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogU))), obs=mp.U)
    else:
        raise ValueError(f"{mp.noisemodel} not allowed")