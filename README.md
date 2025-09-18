# CellSAM: A Foundation Model for Cell Segmentation

**NOTE** Be sure to update to the [latest version of the model][pinned-issue]

[pinned-issue]: https://github.com/vanvalenlab/cellSAM/issues/90

## Description
This repository provides inference code for CellSAM. CellSAM is described in more detail in the [preprint](https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3), and is publicly deployed at [cellsam.deepcell.org](https://cellsam.deepcell.org/). CellSAM achieves state-of-the-art performance on segmentation across a variety of cellular targets (bacteria, tissue, yeast, cell culture, etc.) and imaging modalities (brightfield, fluorescence, phase, etc.). Feel free to [reach out](mailto:ulisrael@caltech.edu) for support/questions! The full dataset used to evaluate CellSAM is available via <https://users.deepcell.org>. This dataset contains cellpose data which is subject to the cellpose license [here](https://github.com/mouseland/cellpose?tab=BSD-3-Clause-1-ov-file). The cellpose data can also be downloaded from [here](https://www.cellpose.org/dataset).

## Getting started
The easiest way to get started with CellSAM is with pip
`pip install git+https://github.com/vanvalenlab/cellSAM.git`

CellSAM requires `python>=3.10`, but otherwise uses pure PyTorch. A sample image is included in this repository. Segmentation can be performed as follows

```
import numpy as np
from cellSAM import segment_cellular_image
img = np.load("sample_imgs/yeaz.npy")
mask, _, _ = segment_cellular_image(img, device='cuda')
```

For more details, see [the tutorial](https://vanvalenlab.github.io/cellSAM/tutorial)

### Napari package
CellSAM includes a basic napari package for annotation functionality. To install the additional napari dependencies, use pip.

```bash
pip install "cellSAM[napari] @ git+https://github.com/vanvalenlab/cellSAM@master"
```

To launch the napari app, run `cellsam napari`.
For further details, check out the [napari plugin docs](https://vanvalenlab.github.io/cellSAM/napari.html)

## Citation

Please cite us if you use CellSAM.

```
@article{israel2023foundation,
  title={A Foundation Model for Cell Segmentation},
  author={Israel, Uriah and Marks, Markus and Dilip, Rohit and Li, Qilin and Schwartz, Morgan and Pradhan, Elora and Pao, Edward and Li, Shenyi and Pearson-Goulart, Alexander and Perona, Pietro and others},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory Preprints},
  doi = {10.1101/2023.11.17.567630},
}
```
