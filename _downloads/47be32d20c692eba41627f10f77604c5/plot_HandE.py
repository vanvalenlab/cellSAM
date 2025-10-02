"""
Histology - H&E Staining
========================

Applying the cellsam_pipeline to segment H&E stained images.

This FOV is extracted from dataset `HBM248.QRTB.362`_ available
on the `HuBMAP data portal`_.

.. _HBM248.QRTB.362: https://portal.hubmapconsortium.org/browse/dataset/692cd314ab251e5b1b1b02a9b7a9beec
.. _HuBMAP data portal: https://portal.hubmapconsortium.org/
"""
import zarr
import skimage
import napari
from cellSAM import cellsam_pipeline
# NOTE: data is stored with zarr_format 3
assert int(zarr.__version__[0]) > 2

# Access H&E image
store = zarr.storage.FsspecStore.from_url(
    "s3://cellsam-gallery-sample-data/sample-data.zarr",
    storage_options={"anon": True},
    read_only=True,
)
z = zarr.open_group(store=store, mode="r")
# Load H&E image into local memory
# Limit to upper-left quadrant to reduce CI computation load
tilesize = 512
img = z["HBM248_QRTB_362"][:tilesize, :tilesize, :]
print(img.shape)

# NOTE: H&E images are often RGB - CellSAM expects RGB images to
# be condensed to a single channel, as with `skimage.color.rgb2gray`,
# for example
mask = cellsam_pipeline(skimage.color.rgb2gray(img), use_wsi=False)

nim = napari.view_image(img, name="H&E image");
nim.add_labels(mask, name="Cellsam segmentation");

if __name__ == "__main__":
    napari.run()
