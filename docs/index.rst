CellSAM Documentation
=====================

Welcome to the CellSAM documentation!

CellSAM is a universal model for cell segmentation.
CellSAM achieves state-of-the-art performance on segmentation across a variety
of cellular targets (bacteria, tissue, yeast, cell culture, etc.) and imaging
modalities (brightfield, fluorescence, phase, multiplexed, etc.).

See the `preprint`_ for more information about the model and the general
approach to image segmentation.

.. _preprint: https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/vanvalenlab/cellSAM.git

.. currentmodule:: cellSAM.cellsam_pipeline

Model Weights
-------------

.. note::
   Pre-trained model weights are available via the deepcell portal at
   https://users.deepcell.org subject to the `license terms`_.

.. _license terms: https://github.com/vanvalenlab/cellSAM/blob/master/LICENSE.md

Pre-trained model weights are required to run the inference pipeline. These will
be automatically downloaded to ``$HOME/.deepcell/models`` the first time you
run :func:`cellsam_pipeline`.

Alternatively, you can call ``get_model`` without any arguments to download the
latest pre-trained model weights.

Basic Usage
-----------

The primary interface is :func:`cellsam_pipeline`, which implements
the inference pipeline for the CellSAM model.
See the :ref:`example_gallery` for examples of CellSAM applied to different
types of images.

Citation
--------

.. code-block:: latex

   @article{israel2023foundation,
     title={A Foundation Model for Cell Segmentation},
     author={Israel, Uriah and Marks, Markus and Dilip, Rohit and Li, Qilin and Schwartz, Morgan and Pradhan, Elora and Pao, Edward and Li, Shenyi and Pearson-Goulart, Alexander and Perona, Pietro and others},
     journal={bioRxiv},
     publisher={Cold Spring Harbor Laboratory Preprints},
     doi = {10.1101/2023.11.17.567630},
   }

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial
   reference/index
   auto_examples/index
   napari

