#!/usr/bin/env python
# coding: utf-8

from math import atan, atan2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class Cycle:
    """
    Represents the parametrization of a cycle in gene expression space using Fourier series expansion.

    Attributes:
    means (pd.DataFrame): Expected values for each coefficient of the Fourier series expansion.
    stds (pd.DataFrame): Standard deviations for each coefficient of the Fourier series expansion.
    log_gammas (np.array): Log degradation rate values, initially None.
    log_betas (np.array): Log splicing rates values, initially None.
    disp_pyro (np.array): Negative binomial disperson values for spliced counts, initially None.
    periodic (np.array): Bernoulli indicate for gene periodicity, initially None.

    Methods:
    __init__: Initialize the Cycle object.
    __len__: Return the length of the Cycle object.
    __getitem__: Enable indexing to access specific elements of the Cycle object.
    set_means: Set new mean values for the coefficients.
    set_stds: Set new standard deviation values for the coefficients.
    set_log_gammas: Set new log gamma values.
    set_log_betas: Set new log beta values.
    set_disp_pyro: Set new dispersion values for Pyro.
    harmonics: Return the number of harmonics in the Fourier series.
    shape: Return the shape of the coefficients matrix.
    load: Class method to load a Cycle object from a file.
    from_file: Class method to load a Cycle object from a file.
    extend: Extend the Cycle object to include new genes.
    add_harmonics: Add additional harmonics to the Fourier series.
    remove_harmonics: Remove a specified number of harmonics from the Fourier series.
    save: Save the Cycle object to a file.
    copy: Create a copy of the Cycle object.
    means_tensor: Return the means as a PyTorch tensor.
    stds_tensor: Return the standard deviations as a PyTorch tensor.
    genes: Return a list of genes represented in the Cycle object.
    from_array: Class method to create a Cycle object from numpy arrays.
    trivial_prior: Class method to create a trivial Cycle object with default parameters.
    polar_plot: Visualize the phases in a polar plot.
    shift_zero: Rotate the phase to a specified gene or phase.
    invert_direction: Invert the direction of the cycle.
    check_orientation: Check if the phase difference between two genes is positive.
    _repr_html_: Provide HTML representation for IPython environments.
    """

    def __init__(self):
        self.means: pd.Dataframe = None
        self.stds: pd.Dataframe = None
        self.log_gammas: np.array = None
        self.log_betas: np.array = None
        self.disp_pyro: np.array = None
        self.periodic: np.array = None
        
    def __len__(self):
        return self.shape[-1]

    def __getitem__(self, key):
        out = type(self)()
        out.means = self.means.__getitem__(key)
        out.stds = self.stds.__getitem__(key)
        return out

    def set_means(self, new_means):
        """
        Set new mean values for the Fourier coefficients.

        Parameters:
        new_means (torch.tensor or np.ndarray or pd.DataFrame): New mean values to be set.

        Raises:
        Exception: If the type of new_means is not recognized.
        """
        if (type(new_means) == torch.tensor):
            new_means = pd.DataFrame(new_means.numpy())
            new_means.index = self.means.index
            new_means.columns = self.means.columns
            self.means = new_means
        elif (type(new_means) == np.ndarray):
            new_means = pd.DataFrame(new_means)
            new_means.index = self.means.index
            new_means.columns = self.means.columns
            self.means = new_means
        elif (type(new_means)==pd.core.frame.DataFrame):
            self.means = new_means
        else:
            raise Exception("Error: invalid type for new_phixy")
            
    def set_stds(self, new_stds):
        """
        Set new standard deviation values for the Fourier coefficients.

        Parameters:
        new_stds (torch.tensor or np.ndarray or pd.DataFrame): New standard deviation values to be set.

        Raises:
        Exception: If the type of new_stds is not recognized.
        """
        if (type(new_stds) == torch.tensor):
            new_stds = pd.DataFrame(new_stds.numpy())
            new_stds.index = self.stds.index
            new_stds.columns = self.stds.columns
            self.stds = new_stds
        elif (type(new_stds) == np.ndarray):
            new_stds = pd.DataFrame(new_stds)
            new_stds.index = self.stds.index
            new_stds.columns = self.stds.columns
            self.stds = new_stds
        elif (type(new_stds)==pd.core.frame.DataFrame):
            self.stds = new_stds
        else:
            raise Exception("Error: invalid type for new_phixy")
    
    def set_log_gammas(self, new_gammas):
        """
        Set new log degradation rate values.

        Parameters:
        new_gammas (np.array): The new log gamma values to set.
        """
        self.log_gammas = new_gammas
    
    def set_log_betas(self, new_betas):
        """
        Set new log splicing rate values.

        Parameters:
        new_betas (np.array): The new log beta values to set.
        """
        self.log_betas = new_betas
    
    def set_disp_pyro(self, new_disp_pyro):
        """
        Set new dispersion values for the negative binomial spliced counts.

        Parameters:
        new_disp_pyro (np.array): The new dispersion values to set.
        """
        self.disp_pyro = new_disp_pyro
    
    @property
    def harmonics(self):
        """
        Get the number of harmonics in the Fourier series.

        Returns:
        int: The number of harmonics.
        """
        return (self.means.shape[0] - 1) // 2

    @property
    def shape(self):
        """
        Get the shape of the coefficients matrix.

        Returns:
        tuple: The shape of the means DataFrame.
        """
        return self.means.shape

    @classmethod
    def load(cls, filepath):
        """
        Load a Cycle object from a file.

        Parameters:
        filepath (str): The path to the file from which to load the Cycle object.

        Returns:
        Cycle: The loaded Cycle object.
        """
        contact_df = pd.read_csv(filepath, index_col=0)
        means, stds = (
            contact_df.iloc[: contact_df.shape[0] // 2, :],
            contact_df.iloc[contact_df.shape[0] // 2 :, :],
        )
        cycle = cls()
        cycle.means = means
        cycle.stds = stds
        return cycle

    @classmethod
    def from_file(cls, filepath):
        """
        Load a Cycle object from a file.

        Parameters:
        filepath (str): The path to the file from which to load the Cycle object.

        Returns:
        Cycle: The loaded Cycle object.
        """
        return cls.load(filepath)

    def extend(self, gene_names, means=0.0, stds=10.0):
        """
        Extend the Cycle object inplace to include new genes.

        Parameters:
        gene_names (list): A list of new gene names to include.
        means (float): Default mean value for new entries. Default is 0.0.
        stds (float): Default standard deviation for new entries. Default is 10.0.
        """
        extension = self.trivial_prior(
            gene_names, harmonics=self.harmonics, means=means, stds=stds
        )
        self.means = pd.concat([self.means, extension.means], axis=1)
        self.stds = pd.concat([self.stds, extension.stds], axis=1)

    def add_harmonics(self, extra_harmonics=1, means=None, stds=None):
        """
        Add additional harmonics to the Fourier series.

        Parameters:
        extra_harmonics (int): The number of additional harmonics to add.
        means (float or np.ndarray): Mean values for the new harmonics.
        stds (float or np.ndarray): Standard deviation values for the new harmonics.
        """
        n = int(self.harmonics)
        for i in range(extra_harmonics):
            N = n + 1 + i
            if means is None:
                self.means.loc[f"nu{N}_cos"] = np.zeros(self.shape[1])
                self.means.loc[f"nu{N}_sin"] = np.zeros(self.shape[1])
            else:
                _means = np.broadcast_to(means, (2 * extra_harmonics, self.shape[1])).copy()
                self.means.loc[f"nu{N}_cos"] = _means[0 + i * 2, :]
                self.means.loc[f"nu{N}_sin"] = _means[1 + i * 2, :]
            if stds is None:
                self.stds.loc[f"nu{N}_cos"] = 10 * np.ones(self.shape[1])
                self.stds.loc[f"nu{N}_sin"] = 10 * np.ones(self.shape[1])
            else:
                _stds = np.broadcast_to(stds, (2 * extra_harmonics, self.shape[1])).copy()
                self.stds.loc[f"nu{N}_cos"] = _stds[0 + i * 2, :]
                self.stds.loc[f"nu{N}_sin"] = _stds[1 + i * 2, :]

    def remove_harmonics(self, n=1):
        """
        Remove a specified number of harmonics from the Fourier series.

        Parameters:
        n (int): The number of harmonics to remove.
        """
        self.means = self.means.iloc[:-n, :]
        self.stds = self.stds.iloc[:-n, :]

    def save(self, pathname):
        """
        Save the Cycle object to a file.

        Parameters:
        pathname (str): The file path where the Cycle object will be saved.
        """
        contact_df = pd.concat([self.means, self.stds])
        contact_df.to_csv(pathname)
    
    def copy(self):
        """
        Create a deep copy of the Cycle object.

        Returns:
        Cycle: A deep copy of the Cycle object.
        """
        return copy.deepcopy(self)

    @property
    def means_tensor(self):
        """
        Get the expected values of the coefficients as a PyTorch tensor.

        Returns:
        torch.Tensor: The means as a PyTorch tensor.
        """
        return torch.tensor(self.means.values.astype(np.float32))

    @property
    def stds_tensor(self):
        """
        Get the standard deviation of the coefficients as a PyTorch tensor.

        Returns:
        torch.Tensor: The standard deviations as a PyTorch tensor.
        """
        return torch.tensor(self.stds.values.astype(np.float32))

    @property
    def genes(self):
        """
        Get a list of genes represented in the Cycle object.

        Returns:
        list: A list of gene names.
        """
        return list(self.means.columns)

    @classmethod
    def from_array(cls, means_array, stds_array, gene_names=None):
        """
        Create a Cycle object from numpy arrays and a set of gene names.

        Parameters:
        means_array (np.ndarray): Array containing the mean values.
        stds_array (np.ndarray): Array containing the standard deviation values.
        gene_names (list): A list of gene names.

        Returns:
        Cycle: The created Cycle object.

        Raises:
        AssertionError: If the shapes of the arrays do not match.
        """
        cycle = cls()
        assert means_array.shape == stds_array.shape, "Shapes of the arrays must be equal"
        if gene_names is not None:
            assert len(gene_names) == means_array.shape[1]
        indexes = ["nu0",] + [
            f"nu{i//2+1}_{'sin' if i % 2 else 'cos'}" for i in range(means_array.shape[0] - 1)
        ]
        cycle.means = pd.DataFrame(means_array, index=indexes, columns=gene_names)
        cycle.stds = pd.DataFrame(stds_array, index=indexes, columns=gene_names)
        return cycle

    @classmethod
    def trivial_prior(cls, gene_names, harmonics=2, means=0.0, stds=3.0):
        """
        Create a trivial Cycle object with specified parameters.

        Parameters:
        gene_names (list): A list of gene names.
        harmonics (int): The number of harmonics in the Fourier series.
        means (float): Default mean value for all coefficients.
        stds (float): Default standard deviation for all coefficients.

        Returns:
        Cycle: The created trivial Cycle object.
        """
        if(harmonics==1): 
            stds=np.array([.1,.2,.2])[:,None]
        if(harmonics==2): 
            stds=np.array([.1,.2,.2,.1,.1])[:,None]

        cycle = cls()
        indexes = [
            "nu0",
        ] + [f"nu{i//2+1}_{'sin' if i % 2 else 'cos'}" for i in range(harmonics * 2)]
        cycle.means = pd.DataFrame(
            np.broadcast_to(means, (harmonics * 2 + 1, len(gene_names))).copy(),
            index=indexes,
            columns=gene_names,
        )
        cycle.stds = pd.DataFrame(
            np.broadcast_to(stds, (harmonics * 2 + 1, len(gene_names))).copy(),
            index=indexes,
            columns=gene_names,
        )
        return cycle
    
    def polar_plot(self, gene_list=None, axes_limits=2):
        """
        Visualize the phases in a polar plot.

        Parameters:
        gene_list (list, optional): A list of genes to include in the plot. If None, all genes are plotted.
        axes_limits (float): The limits for the axes in the plot.
        """
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(30,15))
        ax0.plot(self.means.iloc[0], "o")
        ax0.set_xlabel("gene", size="large")
        ax0.set_ylabel("nu0", size="large")
        
        ax0.xaxis.set_ticks(np.arange(len(self.means.columns)))
        ax0.set_xticklabels(self.means.columns, rotation = 90)
        ax1.add_patch(plt.Circle([0,0], radius=1, color='k', fill=False))
        (x, y) = (self.means.iloc[1], self.means.iloc[2])
        ax1.plot(x, y, "o")
        if gene_list is None:
            gene_list = self.genes
        ax1.scatter([0],[0], c = 'r')
        for i, txt in enumerate(gene_list):
            ix = np.where(np.array(self.genes)==txt)[0][0]
            ax1.annotate(txt, ( x[i],  y[i]+0.02))
        ax1.set_xlabel("nu1_cos", size="large")
        ax1.set_ylabel("nu1_sin", size="large")
        ax1.set_xlim(-axes_limits, axes_limits)
        ax1.set_ylim(-axes_limits, axes_limits)

    def shift_zero(self, gene=None, phase=None):
        """
        Sets the zero by rotating the phase to that of the gene or of a given phase
        """
        if not gene is None:
            if gene in self.means.keys():
                [c,s] = self.means[gene][1 : 3].values
                [c,s] = [c,s]/np.linalg.norm([c,s])
            else:
                raise Exception("Error: gene not found in index")
        elif not phase is None:
            [c, s] = [np.cos(phase), np.sin(phase)]
        else:
            raise Exception("Error: must specify gene or phase for desired shift")
        s = -s
        # rotate all genes & all harmonics
        for g in self.means.keys():
            for i in range(1, 2*self.harmonics+1,  2):
                [c_0, s_0] = self.means[g].iloc[i:i+2]
                self.means[g].iloc[i:i+2] = [ c_0 * c - s_0 * s,  c_0 * s + s_0 * c ] 
                

    def invert_direction(self):
        """
        Invert the direction of the cycle. 
        """
        for g in self.means.keys():
            ind = 2*(1+np.arange(0, self.harmonics))
            self.means[g][ind] = - self.means[g][ind] 

    def check_orientation(self, gene_pair=["TOP2A", "E2F1"]):
        """
        Check if the phase difference between two genes is positive.

        Parameters:
        gene_pair (list): A list containing two gene names.

        Returns:
        bool: True if the phase difference is positive, False otherwise.

        Raises:
        Exception: If invalid gene names are provided.
        """
        genes = self.means.keys()
        [g_1, g_2] = gene_pair
        both = g_1 in genes and g_2 in genes
        if both:
            phi_1 = atan2(self.means[g_1][2], self.means[g_1][1])
            phi_2 = atan2(self.means[g_2][2], self.means[g_2][1])
            if phi_1 < 0: phi_1 = phi_1 + 2*np.pi
            if phi_2 < 0: phi_2 = phi_2 + 2*np.pi
            return (phi_2 - phi_1) > 0
        else:
            raise Exception("Error: invalid gene names")


def reorder(cycle, gene_list):
    """A function to reorder genes in a Cycle object according to a desired order

    Arguments
    ---------
    cycle: object, class Cycle
        Cycle object to sort
    gene_list: np.array
        Array of gene names to use for ordering Cycle object
    
    Returns
    -------
    sorted_cycle: object, class Cycle
        Reordered Cycle object
        
    """
    sorted_cycle = Cycle.from_array(means_array=cycle.means[gene_list], stds_array=cycle.stds[gene_list])
    return sorted_cycle