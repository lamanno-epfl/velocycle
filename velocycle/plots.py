#!/usr/bin/env python
# coding: utf-8

import pyro
from pyro import poutine
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

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