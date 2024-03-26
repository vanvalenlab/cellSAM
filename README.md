# CellSAM: A Foundation Model for Cell Segmentation

[  <img src="https://github.com/rdilip/cellsam_inference/assets/9993319/1f40b5a5-60f1-4980-997d-6059e20d6133" alt="Try the demo!" style="width: 100%;">
](https://cellsam.deepcell.org/)

## Description
This repository provides inference code for CellSAM. CellSAM is described in more detail in the [preprint](https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3), and is publicly deployed at [cellsam.deepcell.org](https://cellsam.deepcell.org/). CellSAM achieves state-of-the-art performance on segmentation across a variety of cellular targets (bacteria, tissue, yeast, cell culture, etc.) and imaging modalities (brightfield, fluorescence, phase, etc.). Feel free to [reach out](mailto:ulisrael@caltech.edu) for support/questions! The full dataset used to train CellSAM is available [here](https://storage.googleapis.com/cellsam-data/dataset.tar.gz).

## Getting started
The easiest way to get started with CellSAM is with pip
`pip install cellSAM`

CellSAM requires `python>=3.10`, but otherwise uses pure PyTorch. A sample image is included in this repository. Segmentation can be performed as follows

```
import numpy as np
from cellSAM import segment_cellular_image
img = np.load("sample_img/yeaz.npy")
mask, _, _ = segment_cellular_image(img, device='cuda')
```

For more details, see `cellsam_introduction.ipynb`.

## Citation

Please cite us if you use CellSAM.

```
@article{israel2023foundation,
  title={A Foundation Model for Cell Segmentation},
  author={Israel, Uriah and Marks, Markus and Dilip, Rohit and Li, Qilin and Schwartz, Morgan and Pradhan, Elora and Pao, Edward and Li, Shenyi and Pearson-Goulart, Alexander and Perona, Pietro and others},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory Preprints}
}
```
