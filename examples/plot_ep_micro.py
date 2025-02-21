"""
Phase Microscopy of Mammalian Cells
===================================
"""
import imageio.v3 as iio
import napari

from cellSAM import cellsam_pipeline
img = iio.imread("../sample_imgs/ep_micro.png")

mask = cellsam_pipeline(
    img,
    low_contrast_enhancement=False,
    use_wsi=False,
    gauge_cell_size=False,
)

nim = napari.view_image(img, name="Phase microscopy image");
nim.add_labels(mask, name="Cellsam segmentation");

if __name__ == "__main__":
    napari.run()
