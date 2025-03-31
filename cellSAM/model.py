import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from warnings import warn

import requests
import os
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

from .sam_inference import CellSAM
from .utils import (
    format_image_shape,
    normalize_image,
    fill_holes_and_remove_small_masks,
    subtract_boundaries,
)


def download_file_with_progress(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(destination, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Something went wrong")

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


def get_model(model: nn.Module = None) -> nn.Module:
    """
    Returns a loaded CellSAM model. If model is None, downloads weights and loads the model with a progress bar.
    """
    cellsam_assets_dir = os.path.join(os.path.expanduser("~"), ".cellsam_assets")
    model_path = os.path.join(cellsam_assets_dir, "cellsam_base_v1.1.pt")

    # path = pkg_resources.resource_filename('my.package', 'resource.dat')

    # config_path = resource_filename(__name__, "modelconfig.yaml")

    # with open(config_path, "r") as config_file:

    config = yaml.safe_load(
        pkgutil.get_data(__name__, "modelconfig.yaml")
    )


    if model is None:
        if not os.path.exists(cellsam_assets_dir):
            os.makedirs(cellsam_assets_dir)
        if not os.path.isfile(model_path):
            print("Downloading CellSAM model weights, please wait...")
            download_file_with_progress(
                "https://storage.googleapis.com/cellsam-data/cellsam_base.pt",
                model_path,
            )
        model = CellSAM(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model


def segment_cellular_image(
    img: np.ndarray,
    model: nn.Module = None,
    normalize: bool = False,
    postprocess: bool = False,
    remove_boundaries: bool = False,
    bounding_boxes: list[list[float]] = None,
    bbox_threshold: float = 0.2,
    fast: bool = False,
    device: str = "cpu",
):
    """
    Args:
        img  (np.array): Image to be segmented with shape (H, W) or (H, W, C)
        model (nn.Module): Loaded CellSAM model. If None, will download weights.
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

    model = get_model(model).eval()
    model.bbox_threshold = bbox_threshold

    img = format_image_shape(img)
    if normalize:
        img = normalize_image(img)
    img = img.transpose((2, 0, 1))  # channel first for pytorch.
    img = torch.from_numpy(img).float().unsqueeze(0)

    if "cuda" in device:
        model, img = model.to(device), img.to(device)

    preds = model.predict(img, x=None, boxes_per_heatmap=bounding_boxes, device=device, fast=fast)
    if preds[0] is None:
        warn("No cells detected in the image.")
        return np.zeros(img.shape[-2:], dtype=np.uint8), None, torch.empty((1, 4))

    segmentation_predictions, _, x, bounding_boxes = preds

    if postprocess:
        segmentation_predictions = postprocess_predictions(segmentation_predictions)

    mask = fill_holes_and_remove_small_masks(segmentation_predictions, min_size=25)
    if remove_boundaries:
        mask = subtract_boundaries(mask)

    return mask, x.cpu().numpy(), bounding_boxes


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
