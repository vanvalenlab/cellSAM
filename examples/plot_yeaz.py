"""
Phase Microscopy: Yeast
=======================
"""
import imageio.v3 as iio
import napari

from cellSAM import cellsam_pipeline
img = iio.imread("../sample_imgs/YeaZ.png")

mask = cellsam_pipeline(img, use_wsi=False)

nim = napari.view_image(img, name="YeaZ");
nim.add_labels(mask, name="Cellsam segmentation");

if __name__ == "__main__":
    napari.run()
