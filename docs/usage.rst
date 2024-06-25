Usage
=====

.. _installation:

Installation
------------

You need to have Python 3.8 or newer installed. All other package versions required are indicated in the requirements.txt file in this repo.

We suggest installing VeloCycle in a separate conda environment, which for example can be created with the command:

.. code-block:: console

   (.venv) $ conda create --name velocycle_env python==3.8

You will probably need to install git next:

.. code-block:: console
   (.venv) $ conda install git

Then you can install VeloCycle using one of the following two approaches:

1. Install the latest release on PyPI:

.. code-block:: console
   (.venv) $ pip install velocycle

2. Install the latest development version:

.. code-block:: console
   (.venv) $ pip install git+https://github.com/lamanno-epfl/velocycle.git@main


