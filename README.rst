# VeloCycle

Manifold-constrained variational inference for RNA velocity of the cell cycle. This is the repository for the VeloCycle framework. Installation instructions and tutorials can be found below.

## Getting started

Please refer to the installation instructions to install VeloCycle. Once you are ready to begin exploring the package, please refer to the [tutorials](https://github.com/lamanno-epfl/velocycle/tree/main/tutorials) contained in this repo. The HTML versions of each tutorial show the expected output of running VeloCycle as well as expected runtimes obtained using a Macbook Pro 2019 edition (faster runtimes will be achieved on newer computers or when using GPUs).

## Installation

You need to have Python 3.8 or newer installed. All other package versions required are indicated in the requirements.txt file in this repo. Installation of VeloCycle should take only a few minutes on a standard operating system.

We suggest installing VeloCycle in a separate conda environment, which for example can be created with the command:

```bash
conda create --name velocycle_env python==3.8
```

You will probably need to install git next:

```bash
conda install git
```

Then you can install VeloCycle using one of the following two approaches:

1. Install the latest release on PyPI:

```bash
pip install velocycle
```

2. Install the latest development version:

```bash
pip install git+https://github.com/lamanno-epfl/velocycle.git@main
```

## Release notes

This is the initial release of VeloCycle corresponding to our preprint: "Statistical inference with a manifold-constrained RNA velocity model uncovers cell cycle speed modulations". 

Link: https://www.biorxiv.org/content/10.1101/2024.01.18.576093v1

These software are still under continuous development.

## Contact

For questions and help requests, you can reach out to [Alex Lederer](mailto:alex.lederer@epfl.ch) and [Gioele La Manno](mailto:gioele.lamanno@epfl.ch). We are happy to hear your feedback and comments!

## Additional materials

For notebooks and data files not used in the tutorials included in this repo, but used in the original publication, please see the following [Google Drive](https://drive.google.com/drive/folders/1G_VPLpD8trPBZ8F8h7cBzcPPkKPn2Fik?usp=drive_link) folder and [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250148). In the future, files will be transferred from the Google Drive to Zenodo.

## Citation

Lederer, A. R., Leonardi, M., Talamanca, L., Herrera, A., Droin, C., Khven, I., Carvalho, H. J. F., Valente, A., Dominguez Mantes, A., Mulet Arabí, P., Pinello, L., Naef, F., & La Manno, G. (2024). Statistical inference with a manifold-constrained RNA velocity model uncovers cell cycle speed modulations. In bioRxiv. https://doi.org/10.1101/2024.01.18.576093
