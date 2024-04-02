import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
import yaml
from pkg_resources import resource_filename

from skimage.morphology import (
    disk,
    binary_opening,
    binary_closing,
    binary_erosion,
    binary_dilation,
)
from scipy.ndimage import gaussian_filter
from segment_anything.utils.amg import remove_small_regions

from .sam_inference import CellSAM
from .utils import (
    format_image_shape,
    normalize_image,
    fill_holes_and_remove_small_masks,
    subtract_boundaries,
)
from ._auth import fetch_data, extract_archive


__all__ = ["segment_cellular_image"]


def get_model(model: nn.Module = None) -> nn.Module:
    """
    Returns a loaded CellSAM model. If model is None, downloads weights and loads the model with a progress bar.
    """
    cellsam_assets_dir = Path.home() / ".deepcell/models"
    model_path = cellsam_assets_dir / "cellsam_base.pt"
    config_path = resource_filename(__name__, 'modelconfig.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    if model is None:
        if not cellsam_assets_dir.exists():
            cellsam_assets_dir.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            fetch_data("models/cellsam_base.tar.gz", cache_subdir="models")
            extract_archive(model_path, cellsam_assets_dir)
            assert model_path.exists()
        model = CellSAM(config)
    model.load_state_dict(torch.load(model_path))
    return model

def segment_cellular_image(
    img: np.ndarray,
    model: nn.Module = None,
    normalize: bool = False,
    postprocess: bool = False,
    remove_boundaries: bool = False,
    bbox_threshold: float = 0.4,
    device: str = 'cpu',
):
    """
    img  (np.array): Image to be segmented with shape (H, W) or (H, W, C)
    model (nn.Module): Loaded CellSAM model. If None, will download weights.
    """
    if 'cuda' in device:
        assert torch.cuda.is_available(), "cuda is not available. Please use 'cpu' as device."

    model = get_model(model).eval()
    model.bbox_threshold = bbox_threshold

    img = format_image_shape(img)
    if normalize:
        img = normalize_image(img)
    img = img.transpose((2, 0, 1))  # channel first for pytorch.
    img = torch.from_numpy(img).float().unsqueeze(0)

    if 'cuda' in device:
        model, img = model.to(device), img.to(device)

    preds = model.predict(img, x=None, boxes_per_heatmap=None, device=device)
    if preds is None:
        print("No cells detected.")
        return None

    segmentation_predictions, _, x, bounding_boxes = preds

    if postprocess:
        segmentation_predictions = postprocess_predictions(segmentation_predictions)

    mask = fill_holes_and_remove_small_masks(segmentation_predictions, min_size=25)
    if remove_boundaries:
        mask = subtract_boundaries(mask)

    return mask, x.cpu().numpy(), bounding_boxes.cpu().numpy()


def postprocess_predictions(mask: np.ndarray):
    mask_values = np.unique(mask)
    new_masks = []
    selem = disk(2)
    for mask_value in mask_values[1:]:
        mask = mask == mask_value
        mask, _ = remove_small_regions(mask, 20, mode="holes")
        mask, _ = remove_small_regions(mask, 20, mode="islands")
        opened_mask = binary_opening(mask, selem)
        closed_mask = binary_closing(opened_mask, selem)
        mask = closed_mask

        selem = disk(10)
        mask = binary_dilation(mask, selem)
        mask = binary_erosion(mask, selem)
        mask = gaussian_filter(mask.astype(np.float32), sigma=3)
        mask = mask > 0.5
        mask = mask.astype(np.uint8) * mask_value
        new_masks.append(mask)

    return np.max(new_masks, axis=0)
