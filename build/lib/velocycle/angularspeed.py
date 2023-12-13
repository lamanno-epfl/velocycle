#!/usr/bin/env python
# coding: utf-8

from math import atan, atan2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class AngularSpeed:
    """
    A class representing the parametrization of angular speed for RNA velocity.

    This class encapsulates means and standard deviations of Fourier series coefficients 
    that represent the angular speed along the cell cycle, i.e. the RNA velocity.

    Attributes:
    ----------
    means : pd.DataFrame
        A DataFrame containing the expected values (means) of each Fourier series coefficient.
    stds : pd.DataFrame
        A DataFrame containing the standard deviations of each Fourier series coefficient.

    Methods:
    ----------
    __init__ : Constructor for the AngularSpeed class.
    __len__ : Returns the length of the AngularSpeed object.
    __getitem__ : Enables indexing and slicing of the AngularSpeed object.
    set_means : Sets new mean values for the coefficients.
    set_stds : Sets new standard deviation values for the coefficients.
    harmonics : Returns the number of harmonics in the Fourier series.
    shape : Returns the shape of the coefficient matrix.
    load : Class method to load an AngularSpeed object from a file.
    from_file : Class method to load an AngularSpeed object from a file (alias for 'load').
    extend : Extends the AngularSpeed object to new conditions.
    add_harmonics : Adds additional harmonics to the Fourier series.
    remove_harmonics : Removes a specified number of harmonics from the Fourier series.
    save : Saves the AngularSpeed object to a file.
    copy : Creates a copy of the AngularSpeed object.
    means_tensor : Returns the means as a PyTorch tensor.
    stds_tensor : Returns the standard deviations as a PyTorch tensor.
    conditions : Returns a list of conditions represented in the AngularSpeed object.
    from_array : Class method to create an AngularSpeed object from numpy arrays.
    trivial_prior : Class method to create a trivial AngularSpeed object.
    """
    
    def __init__(self):
        self.means: pd.Dataframe = None
        self.stds: pd.Dataframe = None

    def __len__(self):
        return self.shape[-1]

    def __getitem__(self, key):
        out = type(self)()
        out.means = self.means.__getitem__(key)
        out.stds = self.stds.__getitem__(key)
        return out

    def set_means(self, new_means):
        """
        Sets new mean values for the Fourier series coefficients.

        Parameters:
        new_means : torch.tensor, np.ndarray, or pd.DataFrame
            The new mean values to be set. Must be compatible with the existing data structure.

        Raises:
        Exception: If the input type for new_means is not supported.
        """
        if (type(new_means) == torch.tensor):
            new_means = pd.DataFrame(new_means)
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
        Sets new standard deviation values for the Fourier series coefficients.

        Parameters:
        new_stds : torch.tensor, np.ndarray, or pd.DataFrame
            The new standard deviation values to be set. Must be compatible with the existing data structure.

        Raises:
        Exception: If the input type for new_stds is not supported.
        """
        if (type(new_stds) == torch.tensor):
            new_stds = pd.DataFrame(new_stds)
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
    
    @property
    def harmonics(self):
        """Gets the number of harmonics"""
        return (self.means.shape[0] - 1) // 2

    @property
    def shape(self):
        """Gets the shape of the nu matrix"""
        return self.means.shape

    @classmethod
    def load(cls, filepath):
        """
        Loads an AngularSpeed object from a specified file.

        Parameters:
        filepath : str
            The path to the file from which to load the AngularSpeed object.

        Returns:
        angularspeed: AngularSpeed
            The loaded AngularSpeed object.

        """
        contact_df = pd.read_csv(filepath, index_col=0)
        means, stds = (
            contact_df.iloc[: contact_df.shape[0] // 2, :],
            contact_df.iloc[contact_df.shape[0] // 2 :, :],
        )
        angularspeed = cls()
        angularspeed.means = means
        angularspeed.stds = stds
        return angularspeed

    @classmethod
    def from_file(cls, filepath):
        """Loads an angularspeed object from a file

        Returns
        -------
        angularspeed: AngularSpeed
            The angularspeed object
        """
        return cls.load(filepath)

    def extend(self, gene_names, means=0.0, stds=3.0):
        """
        Extends the AngularSpeed object to include new conditions.

        Parameters:
        gene_names : list
            A list of new gene names to be added.
        means : float, optional
            The default mean value for new entries. Default is 0.0.
        stds : float, optional
            The default standard deviation for new entries. Default is 3.0.
        """
        extension = self.trivial_prior(
            gene_names, harmonics=self.harmonics, means=means, stds=stds
        )
        self.means = pd.concat([self.means, extension.means], axis=1)
        self.stds = pd.concat([self.stds, extension.stds], axis=1)

    def add_harmonics(self, extra_harmonics=1, means=None, stds=None):
        """
        Adds additional harmonics to the Fourier series.

        Parameters:
        extra_harmonics : int, optional
            The number of additional harmonics to add. Default is 1.
        means : float or array-like, optional
            The mean values for the new harmonics. Default is None.
        stds : float or array-like, optional
            The standard deviation values for the new harmonics. Default is None.
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
        Removes a specified number of harmonics from the Fourier series.

        Parameters:
        n : int
            The number of harmonics to remove.
        """
        self.means = self.means.iloc[:-n, :]
        self.stds = self.stds.iloc[:-n, :]

    def save(self, pathname):
        """
        Saves the AngularSpeed object to a specified file.

        Parameters:
        pathname : str
            The file path where the AngularSpeed object will be saved.
        """
        contact_df = pd.concat([self.means, self.stds])
        contact_df.to_csv(pathname)
    
    def copy(self):
        """
        Creates a deep copy of the AngularSpeed object.

        Returns:
        AngularSpeed
            A deep copy of the AngularSpeed object.
        """
        return copy.deepcopy(self)

    @property
    def means_tensor(self):
        """Gets expected values of mu as a tensor"""
        return torch.tensor(self.means.values.astype(np.float32))

    @property
    def stds_tensor(self):
        """Gets the standard deviation of mu as a tensor"""
        return torch.tensor(self.stds.values.astype(np.float32))

    @property
    def conditions(self):
        """Gets the columns of the object data frame, i.e. the condition names"""
        return list(self.means.columns)

    @classmethod
    def from_array(cls, means_array, stds_array, condition_names=None, Nhω=0):
        """
        Creates an AngularSpeed object from numpy arrays.

        Parameters:
        means_array : np.ndarray
            Array containing the mean values.
        stds_array : np.ndarray
            Array containing the standard deviation values.
        condition_names : list, optional
            A list of condition names. Default is None.
        Nhω : int, optional
            Number of harmonics. Default is 0.

        Returns:
        AngularSpeed
            The created AngularSpeed object.

        Raises:
        AssertionError: If the shapes of the means and standard deviations arrays do not match.
        """
        angularspeed = cls()
        assert means_array.shape == stds_array.shape, "Shapes of the arrays must be equal"
        
        ix = Nhω
            
        indexes = ["nu0",] + [
            f"nu{i//2+1}_{'sin' if i % 2 else 'cos'}" for i in range(ix - 1)
        ]
        
        if len(indexes)==1:
            df = pd.DataFrame([means_array])
        else:
            df = pd.DataFrame(means_array.squeeze()) 
        if len(df.index) == len(indexes):
            df.index = indexes
            df.columns = condition_names
        else:
            df.index=condition_names
            df.columns=indexes
            df = df.T
        angularspeed.means = df
        
        if len(indexes)==1:
            df = pd.DataFrame([stds_array])
        else:
            df = pd.DataFrame(stds_array.squeeze()) 
        if len(df.index) == len(indexes):
            df.index = indexes
            df.columns = condition_names
        else:
            df.index=condition_names
            df.columns = indexes
            df = df.T
        angularspeed.stds = df
        return angularspeed
    
    @classmethod
    def trivial_prior(cls, condition_names, harmonics=1, means=0.0, stds=3.0):
        """
        Creates a trivial AngularSpeed object with specified parameters.

        Parameters:
        condition_names : list
            A list of condition names.
        harmonics : int, optional
            The number of harmonics. Default is 1.
        means : float, optional
            The default mean value for the coefficients. Default is 0.0.
        stds : float, optional
            The default standard deviation for the coefficients. Default is 3.0.

        Returns:
        AngularSpeed
            The created trivial AngularSpeed object.
        """
        angularspeed = cls()
        indexes = [
            "nu0",
        ] + [f"nu{i//2+1}_{'sin' if i % 2 else 'cos'}" for i in range(harmonics * 2)]
        
        Nhω = (2*harmonics)+1
        μω0 = means
        σω0 = stds
        μωi = 0.0
        σωi = 0.05
        μνω = torch.Tensor([μω0,] + [μωi]*(Nhω-1))
        σνω = torch.Tensor([σω0] + [σωi]*(Nhω-1))
        μνω = μνω.repeat(len(condition_names), 1).T
        σνω = σνω.repeat(len(condition_names), 1).T
        
        angularspeed.means = pd.DataFrame(
            np.broadcast_to(μνω, (Nhω, len(condition_names))).copy(),
            index=indexes,
            columns=condition_names,
        )
        angularspeed.stds = pd.DataFrame(
            np.broadcast_to(σνω, (Nhω, len(condition_names))).copy(),
            index=indexes,
            columns=condition_names,
        )
        return angularspeed