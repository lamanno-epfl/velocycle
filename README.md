# VeloCycle

Manifold-constrained variational inference for RNA velocity of the cell cycle. This is the repository for the VeloCycle framework. Installation instructions and tutorials can be found below.

## Getting started

Please refer to the installation instructions below as well as the [tutorials]() contained in this repo.

## Installation

You need to have Python 3.10 or newer installed.

We suggest installing VeloCycle in a separate conda environment, which for example can be created with the command:
```conda create --name velocycle_env python==3.10.9```

1. Install the latest release on PyPI:

```bash
pip install velocycle
```

2. Install the latest development version:

```bash
pip install git+https://github.com/lamanno-epfl/velocycle.git@main
```

You may need to install git first:
```conda install git```

## Release notes

This is the initial release of VeloCycle corresponding to [Lederer et al. 2023](). These software are still under continuous development.

## Contact

For questions and help requests, you can reach out to [Alex Lederer](mailto:alex.lederer@epfl.ch) and [Gioele La Manno](mailto:gioele.lamanno@epfl.ch). We are happy to hear your feedback and comments!
If you found a bug, please use the [issue tracker](https://github.com/lamanno-epfl/velocycle/issues).

## Additional materials

For notebooks and data files not used in the tutorials included in this repo, but used in the original publication, please see the following [Zenodo]() page and [GEO]().

## Citation

The preprint for VeloCycle will be available on bioRxiv soon:

[Statistical inference with a manifold-constrained RNA velocity model uncovers cell cycle speed modulations]()

To cite in your work, please use the citation below:

```
Lederer, A.R., Leonardi, M., Talamanca, L., Herrera, A., Droin, C., Khven, I., Carvalho, Hugo J.F., Valente, A., Dominguez Mantes, A., Mulet Arabi, P., Pinello, L., Naef, F., La Manno, G. Statistical inference with a manifold-constrained RNA velocity model uncovers cell cycle speed modulations. 2023.
```
