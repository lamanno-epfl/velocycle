#!/usr/bin/env python
# coding: utf-8

import pyro
from pyro import poutine
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import pandas as pd
import numpy as np
from .utils import *

def check_model(model, *args):
    """
    Run a Pyro model with tracing and print the shape information of the trace.

    This function clears the Pyro parameter store, runs the specified model with
    the given arguments, and then prints out the shape of each sample in the model
    trace. It's useful for diagnostic purposes to understand the structure and
    behavior of a Pyro model.

    Parameters:
    - model (callable): A Pyro model function to be traced.
    - *args: Variable length argument list for the model function.

    Returns:
    - None: This function prints the trace format shapes and does not return any value.
    """
    pyro.clear_param_store()
    trace = poutine.trace(model).get_trace(*args)
    print(trace.format_shapes())

def live_plot(data_dict, figsize=(16,5), title=''):
    """
    Generate a live plot for visualizing data, typically used in iterative processes like training.

    This function creates a live plot with two subplots. The first subplot displays all iterations,
    and the second focuses on the last 100 iterations. Each entry in the data dictionary is plotted,
    with integer keys resulting in scatter plots and other keys in line plots. The minimum value
    across all data is marked with a horizontal line.

    Parameters:
    - data_dict (dict): A dictionary where keys are labels and values are lists of data points. 
                        Integer keys are treated specially.
    - figsize (tuple of int, optional): The size of the figure. Defaults to (16, 5).
    - title (str, optional): The title of the plot. Defaults to an empty string.

    Returns:
    - None: This function generates a live plot and does not return any value.
    """
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.subplot(121)
    for label, data in data_dict.items():
        if isinstance(label, int):
            plt.scatter(np.full( len(data), label), data,
                        marker="X", s=50, c="r")
        else:
            plt.plot(data, label=label, c="C0")
    plt.axhline(np.min(data), c="r")
    plt.title("All iterations") 
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    
    plt.subplot(122)
    for label,data in data_dict.items():
        if isinstance(label, int):
            continue
        plt.plot(data, label=label, c="C2")
    plt.title("Last 100 iterations") 
    plt.axhline(np.min(data), c="r")
    plt.xlim(len(data)-200, len(data))
    plt.ylim(np.min(data)-10, np.max(data[-200:]))
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.show();

def pplot(phase_fit, show_names=False, show_markers=True, show_genes_from_list=None, species="Human"):
    gene_names = np.array([a.upper() for a in phase_fit.cycle_pyro.means.columns])
    markers = list(S_genes_human) + list(G2M_genes_human)
    phases_list = [S_genes_human, G2M_genes_human, [i.upper() for i in gene_names if i.upper() not in markers]]
    g = []
    gradient = []
    for i in range(len(phases_list)):
        for j in range(len(phases_list[i])):
            g.append(phases_list[i][j])
            gradient.append(i)

    color_gradient_map = pd.DataFrame({'Gene': g,  'Color': gradient}).set_index('Gene').to_dict()['Color']
    colored_gradient = pd.Series(gene_names).map(color_gradient_map)
    
    for i,j in zip(gene_names, colored_gradient):
        if np.isnan(j):
            print(i)
         
    xs = phase_fit.fourier_coef[1]
    ys = phase_fit.fourier_coef[2]
    r = np.log10( np.sqrt(xs**2+ys**2) / phase_fit.fourier_coef_sd[1:, :].sum(0) )
    angle = np.arctan2(xs, ys)
    angle = (angle)%(2*np.pi)
    
    N=50
    width = (2*np.pi) / N
    
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(projection='polar')
    
    # First: only plot dots with a color assignment
    angle_subset = angle[~np.isnan(colored_gradient.values)]
    r_subset = r[~np.isnan(colored_gradient.values)]
    color_subset = colored_gradient.values[~np.isnan(colored_gradient.values)]
    
    # Remove genes with very low expression
    angle_subset = angle_subset[r_subset>=-12]
    color_subset = color_subset[r_subset>=-12]
    gene_names_subset = gene_names[r_subset>=-12]
    r_subset = r_subset[r_subset>=-12]
    
    x=100
    # Take a subset of most highly expressing genes to print the names 
    angle_subset_best = angle_subset[r_subset>np.percentile(r_subset, x)]
    color_subset_best = color_subset[r_subset>=np.percentile(r_subset, x)]
    gene_names_subset_best = gene_names_subset[r_subset>=np.percentile(r_subset, x)]
    r_subset_best = r_subset[r_subset>=np.percentile(r_subset, x)]
    
    # Plot all genes in phases list
    num2color = {0:"tab:orange", 1:"tab:green", 2:"black"}
    ax.scatter(angle_subset, r_subset, c=[num2color[i] for i in color_subset], s=50, alpha=0.3, edgecolor='none', rasterized=True)
    
    # Select and plot on top the genes marking S and G2M traditionally
    angle_subset_markers = angle_subset[color_subset!=2]
    r_subset_markers = r_subset[color_subset!=2]
    gene_names_subset_markers = gene_names_subset[color_subset!=2]
    color_subset_markers = color_subset[color_subset!=2]
    
    ax.scatter(angle_subset_markers, r_subset_markers, c=[num2color[i] for i in color_subset_markers], s=50, alpha=1, edgecolor='none',rasterized=True)
    
    # Annotate genes
    if show_markers:
        for (i, txt), c in zip(enumerate(gene_names_subset_markers), colored_gradient.values):
            ix = np.where(np.array(gene_names_subset_markers)==txt)[0][0]
            ax.annotate(txt[0]+txt[1:].upper(), (angle_subset_markers[ix], r_subset_markers[ix]+0.02))

    if show_names:
        for (i, txt), c in zip(enumerate(gene_names), colored_gradient.values):
            ix = np.where(np.array(gene_names)==txt)[0][0]
            ax.annotate(txt[0]+txt[1:].upper(), (angle[ix], r[ix]+0.02))

    plt.xlim(0, 2*np.pi)
    plt.ylim(-1, )
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],["0", "π/2", "π", "3π/2", "2π"])
    plt.tight_layout()
    plt.show();