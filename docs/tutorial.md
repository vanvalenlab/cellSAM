---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tutorial

This notebook will guide a user through using CellSAM. More details on CellSAM
can be obtained at the preprint: https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3

For biologists, we recommend using {ref}`cellsam_pipeline`, which accepts an
image, automatically downloads weights, and returns a mask. 
{ref}`cellsam_pipeline` has additional functionality (e.g., postprocessing,
normalization, contrast enhancement), but can often be used out of the box.
For machine learning practitioners or users with more esoteric use-cases, we
provide direct access to the model and weights through {func}`get_model`.

For more information or additional assistance, feel free to get in touch.
Please email the following addresses:
```
ulisrael@caltech.edu
mmarks@caltech.edu
rdilip@caltech.edu
qli2@caltech.edu
```

```{code-cell} ipython3
from cellSAM import cellsam_pipeline, get_model
```

```{code-cell} ipython3
import numpy as np
import torch
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
```

## Finding all cells using CellSAM

First, let's see how one can directly use {ref}`cellsam_pipeline` to predict all the
masks in a cell.
We'll load a sample image and pass it through the inference pipeline.
When bounding boxes aren't provided, the `CellFinder` module automatically finds
bounding boxes for all the cells. Notice we specify a GPU device.
Although CellSAM will still work on a CPU, it will be quite slow if there are a
large number of cells. 

```{code-cell} ipython3
# Use GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sample data
img = np.load("yeaz.npy") # H, W, C

# Run inference pipeline
mask = cellsam_pipeline(img, device=str(device), use_wsi=False)

# Visualize results
plt.imshow(mask)
```

# Prompting CellSAM

What if we want to label specific cells?
This is a natural outcome if we use CellSAM as a data engine to accelerate
labeling for new morphologies or cell types.
Let's pick out a specific box and show how we can segment only that cell.
We'll use the model directly.

```{code-cell} ipython3
# Here's the cell we want to segment!
box = [290, 365, 60, 38] # x, y, w, h

rect = patches.Rectangle(
    (box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'
)

plt.figure(figsize=(5, 5))
plt.imshow(img[:, :, -1], cmap='gray')
plt.gca().add_patch(rect)
```

```{code-cell} ipython3
model = get_model()

# We can pass the bounding boxes to the model prediction function
x1, y1, w, h = box
x2, y2 = x1 + w, y1 + h
pred_mask = model.predict(
    img[None].transpose((0, 3, 1, 2)),
    boxes_per_heatmap=[[[x1, x2, y1, y2]]]
)[0]
```

Now, let's visualize the predicted mask.
We'll superimpose the mask as an edge onto our image to see it more clearly

```{code-cell} ipython3
dilated_mask = binary_dilation(pred_mask)
edges = dilated_mask ^ pred_mask

full_img = np.array([img[:, :, -1]] * 3).transpose((1, 2, 0))
r, c = np.where(np.isclose(1.0, edges))
full_img[r, c] = [1,0,0]
plt.imshow(full_img)
```

And that's it!
For more info on CellSAM, feel free to reach out or check out the official
deployment at https://cellsam.deepcell.org.
