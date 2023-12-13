#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pyro.distributions as dist

# Relative imports
from .utils import pack_direction, unpack_direction, torch_fourier_basis


# Definitions to compute analytical variance of a Projected Normal

def _eval_poly(y, coef):
    """
    Evaluate a polynomial with coefficients provided in reverse order.

    Parameters:
    y (float): The input value at which to evaluate the polynomial.
    coef (list): A list of coefficients in reverse order (constant term last).

    Returns:
    float: The result of the polynomial evaluation.
    """
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_COEF_LARGE = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
]
_I1_COEF_SMALL = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.2658733e-1,
    0.301532e-2,
    0.32411e-3,
]
_I1_COEF_LARGE = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    """
    Compute the logarithm of the modified Bessel function of the first kind.

    Parameters:
    x (float or torch.Tensor): The input value(s) at which to evaluate the function.
    order (int): The order of the Bessel function (0 or 1).

    Returns:
    torch.Tensor: The logarithm of the modified Bessel function of the first kind at each input value.
    """
    assert order == 0 or order == 1

    # compute small solution
    y = (x / 3.75).pow(2)
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    mask = x < 3.75
    result = large
    if mask.any():
        result[mask] = small[mask]
    return result


class Phases:
    """
    A class representing the phases, including uncertainty, parametrized as a Projected Normal distribution.

    Attributes:
    phi_xy (pd.DataFrame): DataFrame representing the phases and their uncertainties.
    pcs (np.array): Principal component scores if calculated.
    omegas (np.array): Array of omegas if set.

    Methods:
    set_phixy: Set new phi_xy values.
    set_omegas: Set new omega values.
    phis: Get the phi values as a tensor.
    shape: Get the shape of the phi_xy DataFrame.
    directions: Calculate the directions from phi_xy values.
    concentrations: Calculate the concentrations from phi_xy values.
    stds: Calculate the standard deviations from phi_xy values.
    load: Class method to load Phases object from a file.
    from_file: Class method to load Phases object from a file (alias of 'load').
    save: Save the Phases object to a file.
    phi_xy_tensor: Get phi_xy as a PyTorch tensor.
    from_array: Class method to create a Phases object from numpy arrays.
    from_pca_heuristic: Class method to estimate phases with PCA.
    flat_prior: Class method to create a Phases object with a flat prior.
    shift_zero: Shift the zero point of the phases.
    rotate: Rotate the phases by a given angle.
    invert_direction: Invert the direction of the phases.
    max_corr: Find the maximum correlation with a shift.
    from_cycle_mle: Create Phases object using Maximum Likelihood Estimation from cycle data.
    """

    def __init__(self):
        self.phi_xy: pd.Dataframe = None
        self.pcs: None
        self.omegas: np.array = None
        
    def __len__(self):
        return self.shape[-1]

    def set_phixy(self, new_phixy): 
        """
        Set new values for phi_xy.

        Parameters:
        new_phixy (torch.Tensor or np.ndarray or pd.DataFrame): The new phi_xy values.

        Raises:
        Exception: If the type of new_phixy is invalid.
        """
        if (type(new_phixy) == torch.Tensor):
            new_phixy = pd.DataFrame(new_phixy.numpy())
            new_phixy.index = self.phi_xy.index
            new_phixy.columns = self.phi_xy.columns
            self.phi_xy = new_phixy
        elif (type(new_phixy) == np.ndarray):
            new_phixy = pd.DataFrame(new_phixy)
            new_phixy.index = self.phi_xy.index
            new_phixy.columns = self.phi_xy.columns
            self.phi_xy = new_phixy
        elif (type(new_phixy)==pd.core.frame.DataFrame):
            self.phi_xy = new_phixy
        else:
            raise Exception("Error: invalid type for new_phixy")
    
    def set_omegas(self, new_omegas):
        """
        Set new omega values.

        Parameters:
        new_omegas (np.array): An array of new omega values to set.
        """
        self.omegas = new_omegas
        
    @property
    def phis(self):
        """
        Compute and return phi values as a torch tensor.

        Returns:
        torch.Tensor: A tensor of phi values.
        """
        phis=pack_direction(self.phi_xy_tensor.T)
        phis[phis<0]=phis[phis<0]+2*np.pi
        return phis
    
    @property
    def shape(self):
        """
        Get the shape of the phi_xy DataFrame.

        Returns:
        tuple: The shape of the phi_xy DataFrame.
        """
        return self.phi_xy.shape

    @property
    def directions(self):
        """
        Calculate and return the directions from phi_xy values.

        Returns:
        np.ndarray: An array of direction values computed from phi_xy.
        """
        return np.arctan2(self.phi_xy.values[1, :], self.phi_xy.values[0, :]) % (2 * np.pi)

    @property
    def concentrations(self):
        """
        Calculate and return the concentrations from phi_xy values.

        Returns:
        np.ndarray: An array of concentration values computed from phi_xy.
        """
        return np.sqrt(np.sum(self.phi_xy.values ** 2, 0))

    @property
    def stds(self):
        """
        Calculate and return the standard deviations from phi_xy values.

        Returns:
        np.ndarray: An array of standard deviation values.
        """
        return np.sqrt(
            1
            - (
                _log_modified_bessel_fn(torch.tensor(self.concentrations), order=1)
                - _log_modified_bessel_fn(torch.tensor(self.concentrations), order=0)
            )
            .exp()
            .numpy()
        )

    @classmethod
    def load(cls, filepath):
        """
        Load a Phases object from a file.

        Parameters:
        filepath (str): The path to the file from which to load the Phases object.

        Returns:
        phases: The loaded Phases object.
        """
        df = pd.read_csv(filepath, index_col=0)
        phases = Phases()
        phases.phi_xy = df
        return phases

    @classmethod
    def from_file(cls, filepath):
        """
        Load a Phases object from a file (alias for 'load').

        Parameters:
        filepath (str): The path to the file from which to load the Phases object.

        Returns:
        phases: The loaded Phases object.
        """
        return cls.load(filepath)

    def save(self, pathname):
        """
        Save the Phases object to a file.

        Parameters:
        pathname (str): The file path where the Phases object will be saved.
        """
        self.phi_xy.to_csv(pathname)

    @property
    def phi_xy_tensor(self):
        """
        Return the phi_xy values as a PyTorch tensor.

        Returns:
        torch.Tensor: phi_xy values as a PyTorch tensor.
        """
        return torch.tensor(self.phi_xy.values.astype(np.float32))

    @classmethod
    def from_array(cls, phi_xy_array, cell_names=None):
        """
        Create a Phases object from a numpy array and a set of cell names.

        Parameters:
        phi_xy_array (np.ndarray): A numpy array containing phi_xy values.
        cell_names (list, optional): A list of cell names.

        Returns:
        Phases: The created Phases object.

        Raises:
        AssertionError: If the shape of phi_xy_array does not match the required format.
        """
        phases = Phases()
        assert phi_xy_array.shape[0] == 2, "Shape of the array is incorrect"
        if cell_names is not None:
            assert len(cell_names) == phi_xy_array.shape[1]
        indexes = ["phi_x", "phi_y"]
        phases.phi_xy = pd.DataFrame(phi_xy_array, index=indexes, columns=cell_names)
        return phases

    @classmethod
    def from_pca_heuristic(
        cls,
        anndata_object,
        genes_to_use=None,
        concentration=1.0,
        layer="S_sz",
        small_count=1.0e-1,
        normalize_pcs=True,
        zero_at_min_density=False,
        random_state=0,
        plot=False,
        n_components=2
    ):
        """
        Estimates phases with PCA.

        This method applies PCA to given data and estimates phases based on the principal components.

        Parameters:
        (Various parameters specific to PCA and the data being analyzed)

        Returns:
        Phases: The estimated Phases object.

        Raises:
        ValueError: If the specified layer is not present in the anndata object.
        """

        if layer not in anndata_object.layers:
            raise ValueError(f"{layer=} is not a valid entry anndata.obs")
    
        if (genes_to_use is None):
            X = np.log(anndata_object.layers[layer].T + small_count)
        else:
            X = np.log(anndata_object[:, [i in genes_to_use for i in anndata_object.var.index]].layers[layer].T + small_count)
            
        pca = PCA(n_components, random_state=random_state)
        pcs = pca.fit_transform(X.squeeze().T)

        if normalize_pcs:
            pcts = np.percentile(pcs, [0.5, 99.5, 50], 0)
            pcs = (pcs - pcts[2, :]) / (pcts[1, :] - pcts[0, :])
            nmzd = "Normalized_"
        else:
            nmzd = ""

        angle = np.arctan2(pcs[:, 1], pcs[:, 0]) % (2 * np.pi)
        
        if zero_at_min_density:
            ixsr = np.argsort(angle)
            ixstrt = ixsr[np.diff(angle[ixsr]).argmax() + 1]
            shift = -angle[ixstrt]
            proposed = (angle + shift) % (2 * np.pi)
        else:
            proposed = angle

        # # Adjust the range
        if plot:
            plt.scatter(
                pcs[:, 0], pcs[:, 1], s=1, c=proposed, vmin=0, vmax=2 * np.pi, cmap="hsv"
            )
            plt.xlabel(nmzd + "PC1")
            plt.ylabel(nmzd + "PC2")
            plt.colorbar()
            plt.show()
        phases = cls()
        indexes = ["phi_x", "phi_y"]
        phases.phi_xy = pd.DataFrame(
            np.row_stack([np.cos(proposed), np.sin(proposed)]) * concentration,
            index=indexes,
            columns=anndata_object.obs.index,
        )
        phases.pcs = pcs
        phases.pca = pca
        return phases

    @classmethod
    def flat_prior(cls, anndata_object):
        """
        Create a Phases object with a flat prior.

        Parameters:
        anndata_object (Anndata): An Anndata object containing the necessary data.

        Returns:
        Phases: A Phases object with a flat prior.
        """
        phases = cls()
        indexes = ["phi_x", "phi_y"]
        phases.phi_xy = pd.DataFrame(
            np.zeros((2, anndata_object.shape[0])),
            index=indexes,
            columns=anndata_object.obs.index,
        )
        return phases
    
    def shift_zero(self, gene=None, phase=None):
        """
        Shift the zero point of the phases.

        Parameters:
        gene (str, optional): The gene to shift the phase for. Default is None.
        phase (float, optional): The phase to shift to. Default is None.

        Raises:
        Exception: If neither gene nor phase is specified.
        """
        if not gene is None:
            raise Exception("Error: must phase for desired shift")
        elif not phase is None:
            self.set_phixy(unpack_direction(self.phis-phase).T)
        else:
            raise Exception("Error: must specify gene or phase for desired shift")
            pass

    def rotate(self, angle=None):
        """
        Rotate the phases by a given angle.

        Parameters:
        angle (float, optional): The angle by which to rotate the phases.

        Raises:
        Exception: If angle is not specified.
        """
        if not angle is None:
            [c, s] = [np.cos(angle), np.sin(angle)]
            rot=np.array([[c,-s], [s,c]])
            rotated=np.matmul(rot, self.phi_xy.values)
            self.set_phixy(rotated)
        else:
            raise Exception("Error: must specify angle for desired rotation")
            pass

    def invert_direction(self):
        """
        Invert the direction of the phases.
        """
        flip=np.array([[1.,0.], [0.,-1.]])
        flipped=np.matmul(flip, self.phi_xy.values)
        self.set_phixy(flipped)

    def max_corr(self, counts, npoints=100):
        """
        Find the maximum correlation with a shift in the phases.

        Parameters:
        counts (np.ndarray): Array of count data to correlate with the phases.
        npoints (int): The number of points to use for calculating the shift.

        Returns:
        tuple: Contains the shift with maximum correlation, the correlation value, and the array of correlations.
        """
        shifts=np.arange(0, npoints)/npoints * 2*np.pi
        correlation=[]
        for s in shifts:
            x=self.phis-s
            x[x<0]=x[x<0]+2*np.pi
            c=np.corrcoef(x, counts)[0,1]
            correlation.append(c)
        ind=np.argmax(np.array(correlation))
        return(shifts[ind], correlation[ind], correlation)

    def from_cycle_mle(self, cycle, data, a=1, bins=100, concentration=10., noisemodel='Poisson', dispersion = 0.3):
        """
        Create a Phases object using Maximum Likelihood Estimation from cycle data.

        Parameters:
        cycle (Cycle): A Cycle object containing mean tensor data.
        data (Anndata): An Anndata object containing spliced count data.
        a (float): Scaling factor. Default is 1.
        bins (int): Number of bins for phase estimation. Default is 100.
        concentration (float): Concentration parameter for phase estimation. Default is 10.
        noisemodel (str): The noise model to use ('Poisson' or 'NegativeBinomial'). Default is 'Poisson'.
        dispersion (float): Dispersion parameter for the Negative Binomial model. Default is 0.3.

        Returns:
        Phases: The estimated Phases object.
        """
        fou=cycle.means_tensor
        (nf, ng)=fou.shape
        n_harm=int((nf-1)/2)
        counts=data.obs.n_scounts.values
        nc=counts.shape[0]
        log_counts=torch.tensor(np.log(counts), dtype=torch.float32)
        a = torch.tensor(a)
        log_counts_a = log_counts.unsqueeze(-1).repeat(1,ng) * a
        phis=2*np.pi*torch.arange(0, 1, 1./bins, dtype=torch.float32)
        nn=phis.shape[0]
        b=torch_fourier_basis(phis, num_harmonics=n_harm)
        tmp=torch.matmul(b, fou)
        ElogS = tmp.unsqueeze(-1).repeat(1,1,nc) + log_counts_a.T.unsqueeze(0).repeat(nn,1,1)
        LogS=np.exp(ElogS)
        shape_inv=dispersion
        if noisemodel=='Poisson': d=dist.Poisson(LogS)
        elif noisemodel=='NegativeBinomial': d=dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * LogS))
        else: raise NotImplementedError("Not implemented yet, sorry")
        try: dat=data.layers['spliced'].A.astype(np.int64).T
        except: dat=data.layers['spliced'].astype(np.int64).T
        logP=d.log_prob(torch.tensor(dat)).sum(1)
        phis_mle=phis[torch.argmax(logP, dim=0)]
        self.set_phixy(concentration*unpack_direction(phis_mle).T)
