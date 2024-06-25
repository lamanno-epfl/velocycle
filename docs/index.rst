VeloCycle - statistical RNA velocity inference for the cell cycle
===================================

**VeloCycle** is a module for manifold-constrained variational inference of RNA velocity in the cell cycle. RNA velocity enables the recovery of dynamic information from single-cell RNA-sequencing data using information on unspliced and spliced RNA abundances `La Manno et al. Nature 2018 <https://doi.org/10.1038/s41586-018-0414-6>`_.

Many available RNA velocity algorithms can be fragile and rely on heuristics that lack statistical control. Likewise, the estimated vector field is not dynamically consistent with the traversed gene expression manifold. `VeloCycle <https://github.com/lamanno-epfl/velocycle/>`_ is a generative model of RNA velocity that aims to address these problems by coupling velocity field and manifold estimation in a reformulated, unified framework, so as to coherently identify the parameters of an autonomous dynamical system. Focusing on the cell cycle, VeloCycle can be used to study gene regulation dynamics on one-dimensional periodic manifolds in a statistically robust manner.

VeloCycle can be used to ask fundamental biological questions using RNA velocity, including:
* Is there a statstically significant non-zero cell cycle velocity in my dataset?
* Does cell cycle velocity credibly different between batches, samples, or time points?
* Does the expression of particular genes along the cell cycle change across biological contexts?


VeloCycle's main features
--------

**Manifold-learning** procedure:
* Assign single cells to a continuous cell cycle phase between 0 and 2π
* Identify which genes fluctulate in their expression levels during the cell cycle, and during which phase their expression is at a maximum

**Velocity-learning** procedure:
* Estimate the velocity function (cell and gene independent)
* Convert cell cycle periods to a real-time scale of hours
* Test for statistically significant differences in cell cycle speeds among samples or tissues, both in vitro and in vivo
* Evaluate the credibility of velocity estimates


Contents
--------

.. toctree::

   usage
   api
   tutorial_one_sample
   tutorial_two_samples


Support
--------

This is the initial release of VeloCycle corresponding to our `preprint <https://www.biorxiv.org/content/10.1101/2024.01.18.576093v1>`_: "Statistical inference with a manifold-constrained RNA velocity model uncovers cell cycle speed modulations". These software are still under continuous development.

For questions and help requests, you can reach out to `Alex Lederer <mailto:alex.lederer@epfl.ch>`_ and `Gioele La Manno <mailto:gioele.lamanno@epfl.ch>`_. We are eager to hear your feedback and comments!

Please also visit our `GitHub page <https://github.com/lamanno-epfl/velocycle/>`. For notebooks and data files not used in the tutorials included in this repo, but used in the original publication, please see the following `GoogleDrive <https://drive.google.com/drive/folders/1G_VPLpD8trPBZ8F8h7cBzcPPkKPn2Fik?usp=drive_link>` folder and `GEO <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250148>`. In the future, files will be transferred from the Google Drive to Zenodo with a permanent DOI.


Citation
--------

If you use manifold-constrained RNA velocity with VeloCycle in your research, or would like to learn more about potential applications of our tool, please refer and/or cite the following work:

Lederer, A. R., Leonardi, M., Talamanca, L., Herrera, A., Droin, C., Khven, I., Carvalho, H. J. F., Valente, A., Dominguez Mantes, A., Mulet Arabí, P., Pinello, L., Naef, F., & La Manno, G. (2024). Statistical inference with a manifold-constrained RNA velocity model uncovers cell cycle speed modulations. In bioRxiv. https://doi.org/10.1101/2024.01.18.576093