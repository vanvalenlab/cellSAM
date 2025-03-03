import dask.array as da
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from cellSAM.model import get_model
from skimage.segmentation import relabel_sequential

from cellSAM.model import get_local_model, segment_cellular_image
from cellSAM.utils import relabel_mask, get_median_size, enhance_low_contrast
from cellSAM.wsi import segment_wsi


def use_cellsize_gaging(
        inp,
        model,
        device,
        overlap=200,
        iou_depth=200,
        iou_threshold=0.5,
        bbox_threshold=0.4,
        medium_cell_threshold=0.002,
        tile_size=256,
):
    labels = segment_wsi(inp, overlap, iou_depth, iou_threshold, normalize=False, model=model,
                         device=device, bbox_threshold=bbox_threshold).compute()

    median_size, sizes, sizes_abs = get_median_size(labels)

    print(f"Median size: {median_size:.4f}")

    # only if cells are small we to WSI inference
    if median_size < medium_cell_threshold:
        doing_wsi = True
        # cells are medium or small -> do WSI
        # inp = da.from_array(inp, chunks=tile_size)
        labels = segment_wsi(inp, overlap, iou_depth, iou_threshold, normalize=False, model=model,
                             device=device, bbox_threshold=bbox_threshold).compute()
    else:
        labels = segment_cellular_image(inp, model=model, normalize=False, device=device)[0]

    return labels


def load_image(img, swap_channels=False):
    img = iio.imread(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if swap_channels:
        # switch last 2 channels bc nuclear and whoelcell are switched, #TODO: autmatically detect or have input arg like cellpose
        img = img[..., [0, 2, 1]]
    
    return img

def normalize_image(img):
    # normalize to 0-1 min max - channelwise
    for i in range(3):
        # To accomodate empty channels
        if (np.max(img[..., i]) - np.min(img[..., i])) != 0:
            img[..., i] = (img[..., i] - np.min(img[..., i])) / (np.max(img[..., i]) - np.min(img[..., i]))
        else:
            img[..., i] = img[..., i]
    return img


def cellsam_pipeline(
        img,  # str or np.array
        chunks=256,
        model_path=None,
        bbox_threshold=0.4,
        low_contrast_enhancement=True,
        swap_channels=False,
        use_wsi=True,
        gauge_cell_size=True,
        block_size=400,
        overlap=56,
        iou_depth=56,
        iou_threshold=0.5,
        filter_below_min=True, 
):
    """Run the cellsam inference pipeline on `img`.

    Cellsam is capable of segmenting a variety of cells (bacteria,
    eukaryotic, etc.) spanning all forms of microscopy (brightfield,
    phase, autofluorescence, electron microscopy) and
    staining (H&E, PAS, etc.) / multiplexed (codex, mibi, etc.)
    modalities.

    Parameters
    ----------
    img : array_like with shape ``(W, H)`` or ``(W, H, C)``, where C is 1 or 3
        The image to be segmented. For multiple-channel images, `img` should
        have the following format:

          - **Stained images (e.g H&E)**: ``(W, H, C)`` where ``C == 3``
            representing color channels in RGB format.
          - **Multiplexed images**: ``(W, H, C)`` where ``C == 3`` and the
            channel ordering is: ``(blank, nuclear, membrane)``. The
            ``membrane`` channel is optional, in which case a nuclear segmentation
            is returned.
    chunks : int
        TODO: should this be an option?
    model_path : str or pathlib.Path, optional
        Path to the model weights. If `None` (the default), the latest released
        cellsam generalist model is used.

        .. note:: Downloading the model requires internet access

    bbox_threshold : float in range [0, 1], default=0.4
        TODO: Detail effects of this parameter
    low_contrast_enhancement : bool, default=True
        TODO: Detail effects of this parameter
    swap_channels : bool, default=False
        TODO: this should be removed with loading from file
    use_wsi : bool, default=True
        Whether to use tiling to support large images, default is True.
        Generally, tiling is not required when there are fewer than ~3000
        cells in an image.
    gauge_cell_size : bool, default=True
        TODO: Detail effects of this parameter
    block_size : int
        Size of the tiles when `use_wsi` is `True`. In practice, should
        be in the range ``[256, 2048]``, with smaller tile sizes
        preferred for dense (i.e. many cells/FOV) images.
    overlap : int
        Tile overlap region in which label merges are considered. Must
        be smaller than `block_size`. For reliable tiling, value should
        be large enough to encompass `iou_threshold` of the extent of
        a typical object.
    iou_depth : int
        TODO: Detail effects of this parameter: is this/should this be
        distinct from overlap?
    filter_below_min : bool
        TODO: Detail this parameter - is it necessary?

    Returns
    -------
    segmentation_mask : 2D numpy.ndarray of dtype `numpy.uint32`
        A `numpy.ndarray` representing the segmentation mask for `img`.
        The array is 2D with the same dimensions as `img`, with integer
        labels representing pixels corresponding to cell instances.
        Background is denoted by ``0``.

    Examples
    --------
    Using CellSAM to segment a slice from the `~skimage.data.cells3d` dataset.

    >>> import numpy as np
    >>> import skimage
    >>> data = skimage.data.cells3d()
    >>> data.shape
    (60, 2, 256, 256)

    From the `~skimage.data.cells3d` docstring, ``data`` is a 3D multiplexed
    image with dimensions ``(Z, C, X, Y)`` where the ordering of the channel
    dimension ``C`` is ``(membrane, nuclear)``.
    Start by extracting a 2D slice from the 3D volume. The middle slice is
    chosen arbitrarily:

    >>> img = data[30, ...]

    For multiplexed images, CellSAM expects the channel ordering to be
    ``(blank, nuclear, membrane)``:

    >>> seg = np.zeros((*img.shape[1:], 3), dtype=img.dtype)
    >>> seg[..., 1] = img[1, ...]  # nuclear channel
    >>> seg[..., 2] = img[0, ...]  # membrane channel

    Segment the image with `cellsam_pipeline`. Since this is a small image,
    we'll set ``use_wsi=False``. We'll also forgo any pre/post-processing
    by deactivating `low_contrast_enhancement` and `gauge_cell_size`:

    >>> mask = cellsam_pipeline(
    ...     seg, low_contrast_enhancement=False, use_wsi=False, gauge_cell_size=False
    ... )

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_path is not None:
        modelpath = model_path
        model = get_local_model(modelpath)
        model.bbox_threshold = bbox_threshold
        model = model.to(device)
    else:
        model = None
        
        # To prevent creating model for each block
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = get_model(None)
        model = model.to(device)
        model.eval()
        

    if isinstance(img, str):
        img = load_image(img, swap_channels=swap_channels)

    img = img.astype(np.float32)

    img = normalize_image(img) 

    if low_contrast_enhancement:
        img = enhance_low_contrast(img)

    inp = da.from_array(img, chunks=chunks)

    if use_wsi:
        if gauge_cell_size:
            labels = use_cellsize_gaging(inp, model, device)
        else:
            labels = segment_wsi(inp, block_size, overlap, iou_depth, iou_threshold, normalize=False, model=model,
                                 device=device, bbox_threshold=bbox_threshold).compute()
    else:
        labels = segment_cellular_image(inp, model=model, normalize=False, device=device)[0]

    return labels


