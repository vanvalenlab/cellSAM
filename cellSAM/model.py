import torch
import torch.nn as nn
import numpy as np

from warnings import warn

import requests
import os
from pathlib import Path
import yaml
import pkgutil
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
from typing import List

from .sam_inference import CellSAM
from .utils import (
    format_image_shape,
    normalize_image,
    fill_holes_and_remove_small_masks,
    subtract_boundaries,
)
from . import _auth


__all__ = ["segment_cellular_image"]


def get_local_model(model_path: str) -> nn.Module:
    """
    Returns a loaded CellSAM model from a local path.
    """
    config_path = resource_filename(__name__, 'modelconfig.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model = CellSAM(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    return model


def get_model(model="cellsam_general", version=None) -> nn.Module:
    """
    Returns a loaded CellSAM model.

    If pretrained model weights specified by ``version`` are not found locally,
    they will be downloaded from users.deepcell.org.

    Parameters
    ----------
    model : str, default="cellsam_optimized"
       Which model to load. Options include:

        - ``"cellsam_general"``
        - ``"cellsam_optimized"``

       ``"cellsam_general"`` is trained only on publicly-available datasets and
       is made available for reproducibility. Use this model e.g. to reproduce
       the model evaulation cited in the publication.

       ``"cellsam_optimized"`` incorporates additional non-public training data
       and therefore is recommended for the standard use-case (i.e. applying
       to new images).

    version : str, optional. Default=latest
       Which version of the model to use. When ``version=None`` (the default),
       the latest released version will be used.
       
    """
    version = "1.2" if version is None else version
    if version not in _auth._model_versions:
        raise ValueError(
            f"Model version {version} not recognized, must be one of:\n"
            f"{list(_auth._model_versions)}"
        )
    record = _auth._model_versions[version]
    archive_name = record["asset_key"].split("/")[-1]

    cellsam_assets_dir = Path.home() / f".deepcell/models"
    model_version_dir = cellsam_assets_dir / f"cellsam_v{version}"
    model_path = model_version_dir / f"{model}.pt"

    config_path = resource_filename(__name__, 'modelconfig.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    if not cellsam_assets_dir.exists():
        cellsam_assets_dir.mkdir(parents=True)

    # If the version-specific directory does not exist, that means the model
    # weights for the requested version have not been downloaded
    if not model_version_dir.exists():
        _auth.fetch_data(
            record["asset_key"], cache_subdir="models", file_hash=record["asset_hash"]
        )
        _auth.extract_archive(
            cellsam_assets_dir / archive_name, path=cellsam_assets_dir
        )

    model = CellSAM(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model


def segment_cellular_image(
    img: np.ndarray,
    model: nn.Module,
    normalize: bool = True,
    postprocess: bool = False,
    remove_boundaries: bool = False,
    bounding_boxes: List[List[float]] = None,
    bbox_threshold: float = 0.4,
    fast: bool = False,
    device: str = "cpu",
):
    """
    Args:
        img  (np.array): Image to be segmented with shape (H, W) or (H, W, C)
        model (nn.Module): Loaded CellSAM model.
        normalize (bool): If True, normalizes the image using percentile thresholding and CLAHE.
        postprocess (bool): If True, performs custom postprocessing on the segmentation mask. Recommended for noisy images.
        remove_boundaries (bool): If True, removes a one pixel boundary around the segmented cells.
        bounding_boxes (list[list[float]]): List of bounding boxes to be used for segmentation in format
            (x1, y1, x2, y2). If None, will use the model's predictions.
        bbox_threshold (float): Threshold for bounding box confidence.
        fast (bool): Whether or not to use batched inference. Batched inference can be several times faster than standard
            inference, but is an alpha feature and may lead to slightly different results.
        device: 'cpu' or 'cuda'. If 'cuda' is selected, will use GPU if available.
    Returns:
        mask (np.array): Integer array with shape (H, W)
        x (np.array | None): Image embedding
        bounding_boxes (np.array | None): list of bounding boxes
    """
    if "cuda" in device:
        assert (
            torch.cuda.is_available()
        ), "cuda is not available. Please use 'cpu' as device."
    if bounding_boxes is not None:
        bounding_boxes = torch.tensor(bounding_boxes).unsqueeze(0)
        assert (
            len(bounding_boxes.shape) == 3
        ), "Bounding boxes should be of shape (number of boxes, 4)"

    model = model.eval()
    model.bbox_threshold = bbox_threshold

    img = format_image_shape(img)
    if normalize:
        img = normalize_image(img)
    img = img.transpose((2, 0, 1))  # channel first for pytorch.
    img = torch.from_numpy(img).float().unsqueeze(0)

    if "cuda" in device:
        model, img = model.to(device), img.to(device)

    preds = model.predict(img, boxes_per_heatmap=bounding_boxes)
    if preds is None:
        warn("No cells detected in the image.")
        return np.zeros(img.shape[1:], dtype=np.int32), None, None

    segmentation_predictions, _, x, bounding_boxes = preds

    if postprocess:
        segmentation_predictions = postprocess_predictions(segmentation_predictions)

    mask = fill_holes_and_remove_small_masks(segmentation_predictions, min_size=25)
    if remove_boundaries:
        mask = subtract_boundaries(mask)

    return mask, x.cpu().numpy(), bounding_boxes


def postprocess_predictions(all_masks: np.ndarray):
    mask_values = np.unique(all_masks)
    new_masks = []
    selem = disk(2)
    for mask_value in mask_values[1:]:
        mask = all_masks == mask_value
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
