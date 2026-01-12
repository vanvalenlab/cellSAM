"""
Electron Microscopy
===================

Applying the cellsam_pipeline to segment electron microscopy
images of bacteria.

This dataset comprises an electron microscope image of a bacterial
biofilm.
This FOV is downsampled from data provided by the
`Newman Lab at Caltech`_; image credit Mark Ladinsky @ the Caltech
CryoEM facility.

.. _Newman Lab at Caltech: https://dknweb.caltech.edu/
"""
import zarr
import skimage
import napari
from cellSAM import cellsam_pipeline
# NOTE: data is stored with zarr_format 3
assert int(zarr.__version__[0]) > 2

# Access EM image
store = zarr.storage.FsspecStore.from_url(
    "s3://cellsam-gallery-sample-data/sample-data.zarr",
    storage_options={"anon": True},
    read_only=True,
)
z = zarr.open_group(store=store, mode="r")
# Load EM image into local memory
# Limit to lower-right quadrant to reduce CI computation load
tilesize = 512
img = z["biofilm_electron_microscopy"][tilesize:, tilesize:]
print(img.shape)

# Segment
mask = cellsam_pipeline(img, use_wsi=False)

# Visualize
nim = napari.view_image(img, name="EMimage");
nim.add_labels(mask, name="Cellsam segmentation");

if __name__ == "__main__":
    napari.run()
