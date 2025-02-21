"""
Multiplexed Imaging: CODEX
==========================
"""
import imageio.v3 as iio
import napari

from cellSAM.cellsam_pipeline import cellsam_pipeline
img = iio.imread("../sample_imgs/tissuenet.png")

# Image is 3-channel RGB where Channel 1 (G) represents a nuclear stain
# and Channel 2 (B) a membrane stain. Channel 0 (R) is blank.
print(img.sum(axis=(0, 1)))

mask = cellsam_pipeline(
    img,
    low_contrast_enhancement=False,
    use_wsi=False,
    gauge_cell_size=False,
)

nim = napari.view_image(img, name="CODEX image");
nim.add_labels(mask, name="Cellsam segmentation");

if __name__ == "__main__":
    napari.run()
