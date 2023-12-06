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
from pyro.infer import Trace_ELBO, SVI, Predictive, TraceEnum_ELBO, config_enumerate
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoNormal, AutoGuideList
from pyro.infer.autoguide import init_to_mean, init_to_median
import pyro.distributions as dist

# VeloCycle imports
from .plots import live_plot, pplot
from .cycle import Cycle
from .phases import Phases
from .utils import (
    pack_direction,
    torch_fourier_basis,
    torch_basis,
)

def invert_direction(cycle, phases):
    """
    Inverts the direction of cycle and phase data.

    This function takes two objects, 'cycle' and 'phases', and calls their respective 
    'invert_direction' methods to invert their direction data.

    Parameters:
    - cycle: An object representing the cycle data.
    - phases: An object representing the phase data.

    Returns:
    - None
    """
    cycle.invert_direction()
    phases.invert_direction()

def shift_zero(cycle, phases, metaparameters, gene=None, phase=None):
    """
    Shifts the zero point in cycle and phase objects based on a specified gene or phase.

    If a gene is specified, it finds the phase corresponding to the maximum expression of that gene
    and shifts the zero point to that phase. If a phase is specified, both the cycle and phase objects
    are shifted to that phase.

    Parameters:
    - cycle: An object representing the cycle data.
    - phases: An object representing the phase data.
    - metaparameters: An object containing metaparameters for the model.
    - gene (optional): The gene to base the shift on.
    - phase (optional): The phase to shift to.

    Returns:
    - None

    Raises:
    - Exception: If neither gene nor phase is specified.
    """
    if not gene is None:        
        zeta = torch_basis(phases.phis.squeeze(), der=0, kind=metaparameters.basis_kind, **metaparameters.kwargsζ)
        ElogS_before = (cycle.means_tensor.T.unsqueeze(-2) * zeta).sum(-1)
        cycle.shift_zero(gene=gene)
        max_ix_before = np.argmax(ElogS_before.squeeze()[np.where(np.array(cycle.genes)==gene)[0][0], :])
        phase_shift = phases.phis[max_ix_before]
        phases.shift_zero(phase=phase_shift)
    elif not phase is None:
        cycle.shift_zero(phase=phase)
        phases.shift_zero(phase=phase)
    else:
        raise Exception("Error: must specify gene or phase for desired shift")


class PhaseFitModel:
    """
    A class for executing manifold-learning of the cell cycle phases and gene harmonics.

    This class encapsulates the model, guide, and fitting process for phase data. It supports conditioning on 
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
        self.get_posterior = True
        self.num_samples = num_samples
        self.n_per_bin = n_per_bin

    def fit(
        self,
        optimizer,
        loss=pyro.infer.Trace_ELBO(num_particles = 1),
        num_steps=1000,
        intermediate_output_step_size=100,
        store_output=False,
        verbose=True,
    ):
        """
        Fits the model to the data using Stochastic Variational Inference (SVI).

        Parameters:
        - optimizer: The optimizer to use for SVI.
        - loss (optional): The loss function for SVI. Defaults to Trace_ELBO with one particle.
        - num_steps (int, optional): Number of steps to run SVI. Defaults to 1000.
        - intermediate_output_step_size (int, optional): Steps interval for intermediate output. Defaults to 100.
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
        for step in range(num_steps):
            loss = svi.step(self.metaparams)
            losses.append(loss)
            if store_output:
                if step % intermediate_output_step_size == 0:
                    intermediate_output.append(self.sample_posterior(num_samples=50, take_mean=False))
                    logging.info("Elbo loss: {}".format(loss))

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
        
        self.phis_pyro = pyro.param("ϕxy_locs").detach().squeeze().cpu().numpy().T
        self.fourier_coef = pyro.param("ν_locs").detach().squeeze().cpu().numpy().T
        self.fourier_coef_sd = pyro.param("ν_scales").detach().squeeze().cpu().numpy().T

        new_cycle = Cycle.from_array(self.fourier_coef, self.fourier_coef_sd, self.metaparams.cycle_prior.genes)
        new_phase = Phases.from_array(self.phis_pyro, cell_names=self.metaparams.phase_prior.phi_xy.columns)
        
        self.disp_pyro = pyro.param("shape_inv_locs").detach().squeeze().cpu().numpy().T
        if self.metaparams.with_delta_nu:
            self.delta_nus = pyro.param("Δν_locs").detach().unsqueeze(-3).unsqueeze(-4).float().cpu().numpy()
        
        self.cycle_pyro = new_cycle
        self.phase_pyro = new_phase

        # Put estimations in new objects
        if self.get_posterior:
            if False:
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
    
                phase_pps_cpu_dict_list = []
                for curr_bin in range(nbins):
                    if (self.metaparams.gene_selection_model=="lba"):
                        if self.metaparams.with_delta_nu:
                            phase_pps = self.sample_posterior(num_samples=n_per_bin, rs=['ν', 'Δν', 'ϕxy', 'shape_inv', 'ϕ', 'ζ', 'periodic', 'periodic_prob'])
                        else:
                            phase_pps = self.sample_posterior(num_samples=n_per_bin, rs=['ν', 'ϕxy', 'shape_inv', 'ϕ', 'ζ', 'periodic', 'periodic_prob'])
                    else:
                        if self.metaparams.with_delta_nu:
                            phase_pps = self.sample_posterior(num_samples=n_per_bin, rs=['ν', 'Δν', 'ϕxy', 'shape_inv', 'ϕ', 'ζ'])
                        else:
                            phase_pps = self.sample_posterior(num_samples=n_per_bin, rs=['ν', 'ϕxy', 'shape_inv', 'ϕ', 'ζ'])
                    phase_pps_cpu = {k: v.cpu() for k, v in phase_pps.items()}  # Move samples to CPU
                
                    del phase_pps
                    torch.cuda.empty_cache()
    
                    phase_pps_cpu_dict_list.append(phase_pps_cpu)

                phase_pps_cpu_full = {}
                for k in phase_pps_cpu_dict_list[0].keys():
                    concat_pps_tensor = torch.vstack([phase_pps_cpu_dict_list[i][k] for i in range(nbins)])
                    phase_pps_cpu_full[k] = concat_pps_tensor

                ν = pyro.param("ν_locs").detach().cpu()
                if self.metaparams.with_delta_nu:
                    Δν = pyro.param("Δν_locs").detach().cpu()
                
                ζ = torch_fourier_basis(self.phase_pyro.phis, num_harmonics=self.metaparams.num_harmonics_S, der=0, device=torch.device("cpu")) 
                
                if self.metaparams.with_delta_nu:
                    ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + torch.einsum("bgc,bgc->gc", self.metaparams.Db.to(torch.device("cpu")), Δν) + self.metaparams.count_factor.to(torch.device("cpu"))
                    ElogS2 = torch.einsum("...gch,ch->gc", ν, ζ) + torch.einsum("bgc,bgc->gc", self.metaparams.Db.to(torch.device("cpu")), Δν) + self.metaparams_avg.count_factor.to(torch.device("cpu"))

                else:
                    ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + self.metaparams.count_factor.to(torch.device("cpu"))
                    ElogS2 = torch.einsum("...gch,ch->gc", ν, ζ) + self.metaparams_avg.count_factor.to(torch.device("cpu"))

                phase_pps_cpu_full["ElogS"] = ElogS.squeeze()
                phase_pps_cpu_full["ElogS2"] = ElogS2.squeeze()
                
                self.posterior = phase_pps_cpu_full

                if (self.metaparams.gene_selection_model=="lba"):
                    self.periodic = self.posterior["periodic"].mean(0).squeeze().cpu().numpy()
                    self.periodic_probs = pyro.param("logit_locs").detach().squeeze().cpu().numpy().T
                
        if store_output:
            return intermediate_output

    def sample_posterior(self, num_samples=1, rs=None, mp=None, take_mean=True):
        """
        Samples from the posterior distribution of the model.
    
        Parameters:
        - num_samples (int, optional): The number of samples to draw from the posterior. Defaults to 1.
        - rs (list, optional): A list of sites to return in the posterior sample. If None, all sites are returned.
        - mp (dict, optional): Metaparameters for the model. If None, uses the object's metaparams attribute.
        - take_mean (bool, optional): If True, the mean of the samples is taken. Currently unused in the function.
    
        Returns:
        - dict: A dictionary of posterior samples for each site, moved to CPU memory.
    
        This function generates posterior samples using the Predictive class from Pyro, allowing for sampling
        from the posterior distribution of the model's parameters. It optionally takes specific return sites and 
        metaparameters. All samples are moved to CPU memory before being returned.
        """
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
        """
        Utility function for checking the structure and shape of a Pyro model.
    
        Parameters:
        - m (function): The Pyro model to be checked.
        - args: Variable length argument list for the model 'm'.
    
        This function clears the Pyro parameter store, traces the provided model 'm' with the provided arguments,
        and prints the format of the trace with shapes. It's used for debugging purposes to understand the
        model's structure and the shape of its distributions and parameters.
        """
        pyro.clear_param_store()
        trace = poutine.trace(m).get_trace(*args)
        print(trace.format_shapes())

    def check_model(self):
        """
        Checks the structure and shape of the object's model.
    
        This method is a convenience wrapper around the `_check_model` function. It uses the object's model and
        metaparameters to perform the check. This is useful for quickly verifying the structure and shape of 
        the model associated with the object.
        """
        return self._check_model(self.model, self.metaparams)

    def check_guide(self):
        """
        Checks the structure and shape of the object's guide (variational distribution).
    
        This method is a convenience wrapper around the `_check_model` function. It uses the object's guide and
        metaparameters to perform the check. This is useful for quickly verifying the structure and shape of 
        the guide associated with the object, which is essential in variational inference.
        """
        return self._check_model(self.guide, self.metaparams)       

    def polar_plot(self, show_names=False, show_markers=True, species="Human"):
        pplot(self, show_names=False, show_markers=True, species=species)
        
def phase_latent_variable_model(mp):
    """
    Defines the Pyro model for a phase latent variable.

    Parameters:
    - mp: An object containing model parameters and device information.

    Returns:
    - None
    """
    device = mp.device  # Assuming mp object has an attribute specifying the device
    
    # Plates initialization
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-3, device=device)

    with gene_plate:
        ν = pyro.sample("ν", dist.Normal(mp.μνg.to(device), mp.σνg.to(device)).to_event(1))
        with batches_plate:
            if mp.with_delta_nu:
                Δν = pyro.sample("Δν", dist.Normal(0, mp.σΔν.to(device)))
    with cell_plate:
        ϕxy = pyro.sample("ϕxy", dist.Normal(mp.ϕxy_prior.to(device), torch.tensor(1.0, device=device)).to_event(1))

    ϕ = pack_direction(ϕxy)
    ζ = torch_fourier_basis(ϕ.squeeze(), num_harmonics=mp.num_harmonics_S, der=0, device=device) 

    pyro.deterministic("ϕ", ϕ)
    pyro.deterministic("ζ", ζ)

    if mp.with_delta_nu:
        ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + torch.einsum("bgc,bgc->gc", mp.Db.to(device), Δν) + mp.count_factor.to(device)
    else:
        ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + mp.count_factor.to(device)
           
    pyro.deterministic("ElogS", ElogS)

    # Add noise to the expectation
    if mp.noisemodel == "Lognormal":
        with gene_plate, cell_plate:
            pyro.sample("logS", dist.Normal(ElogS, mp.σgc.to(device)), obs=mp.logS.to(device))
        # Noise
    elif mp.noisemodel == "Poisson":
        with cell_plate, gene_plate:
            pyro.sample("S", dist.Poisson(torch.exp(ElogS)), obs=mp.S.to(device))
    elif mp.noisemodel == "NegativeBinomial":
        with gene_plate:
            shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        with cell_plate, gene_plate:
            pyro.sample("S",dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogS))), obs=mp.S.to(device))
    else:
        raise ValueError(f"{mp.noisemodel} not allowed")

@config_enumerate
def phase_latent_variable_model_lba(mp):
    """
    Defines a Latent Bernoulli Allocation (LBA) model for phase latent variables in Pyro.

    Similar to 'phase_latent_variable_model', but incorporates a LBA approach, adding sampling 
    of 'periodic_prob' and 'periodic' to handle gene periodicity and select periodic vs. non-periodic genes.

    Parameters:
    - mp: An object containing model parameters and device information.

    Returns:
    - None
    """
    device = mp.device  # Assuming mp object has an attribute specifying the device
    
    # Plates initialization
    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device)
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device)
    batches_plate = pyro.plate("batches", mp.Nb, dim=-3, device=device)

    with gene_plate:
        ν = pyro.sample("ν", dist.Normal(mp.μνg.to(device), mp.σνg.to(device)).to_event(1))
        with batches_plate:
            if mp.with_delta_nu:
                Δν = pyro.sample("Δν", dist.Normal(0, mp.σΔν.to(device)))
                
        periodic_prob = pyro.sample('prob', dist.Beta(mp.beta0.to(device), mp.beta1.to(device)))
        periodic = pyro.sample('periodic', dist.Bernoulli(periodic_prob), infer={"enumerate": "parallel"})
        
    with cell_plate:
        ϕxy = pyro.sample("ϕxy", dist.Normal(mp.ϕxy_prior, 1.0).to_event(1))

    ϕ = pack_direction(ϕxy)
    ζ = torch_fourier_basis(ϕ.squeeze(), num_harmonics=mp.num_harmonics_S, der=0, device=device) 

    pyro.deterministic("ϕ", ϕ)
    pyro.deterministic("ζ", ζ)

    ElogS = torch.where(periodic==1,
                    (ν * ζ).sum(-1) + (mp.Db * Δν).sum(-3) + mp.count_factor.to(device), 
                    (ν[:, :, 0] * ζ[:, 0]) + (mp.Db * Δν).sum(-3) + mp.count_factor.to(device))
                    
    pyro.deterministic("ElogS", ElogS)

    # Add noise to the expectation
    if mp.noisemodel == "Lognormal":
        with gene_plate, cell_plate:
            pyro.sample("logS", dist.Normal(ElogS, mp.σgc.to(device)), obs=mp.logS.to(device))
        # Noise
    elif mp.noisemodel == "Poisson":
        with cell_plate, gene_plate:
            pyro.sample("S", dist.Poisson(torch.exp(ElogS)), obs=mp.S.to(device))
    elif mp.noisemodel == "NegativeBinomial":
        with gene_plate:
            shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        with cell_plate, gene_plate:
            pyro.sample("S",dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogS))), obs=mp.S.to(device))
    else:
        raise ValueError(f"{mp.noisemodel} not allowed")