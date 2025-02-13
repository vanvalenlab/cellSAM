import numpy as np
from scipy.ndimage import binary_fill_holes, find_objects
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from scipy.stats import hmean


def enhance_low_contrast(img,
                         lower_contrast_threshold=0.05,
                         upper_contrast_threshold=0.1,
                         max_green_channel_value=0,
                         clip_limit_default=0.01,
                         kernel_size_default=256,
                         gamma_default=2,
                         clip_limit_high_diff=0.02,
                         kernel_size_high_diff=384,
                         gamma_high_diff=1.2,
                         bbox_threshold_high_diff=0.15,
                         clip_limit_very_high_diff=0.05,
                         bbox_threshold_very_high_diff=0.15,
                         clip_limit_adjusted=0.01,
                         std_range=(0.035, 0.04),
                         mean_diff_threshold=0.065,
                         mean_std_threshold=0.05):
    """
    Enhance low contrast images using CLAHE and gamma adjustment.

    Parameters:
        img (ndarray): Input image.
        lower_contrast_threshold (float): Lower threshold for contrast classification.
        upper_contrast_threshold (float): Upper threshold for contrast classification.
        max_green_channel_value (int): Maximum allowed value in the green channel.
        clip_limit_default (float): Default CLAHE clip limit.
        kernel_size_default (int): Default CLAHE kernel size.
        gamma_default (float): Default gamma adjustment value.
        clip_limit_high_diff (float): CLAHE clip limit for high mean_diff.
        kernel_size_high_diff (int): CLAHE kernel size for high mean_diff.
        gamma_high_diff (float): Gamma adjustment value for high mean_diff.
        bbox_threshold_high_diff (float): Bbox threshold for high mean_diff.
        clip_limit_very_high_diff (float): CLAHE clip limit for very high mean_diff.
        bbox_threshold_very_high_diff (float): Bbox threshold for very high mean_diff.
        clip_limit_adjusted (float): Adjusted CLAHE clip limit for specific std range.
        std_range (tuple): Range of mean_std for specific adjustments.
        mean_diff_threshold (float): Threshold for very high mean_diff.
        mean_std_threshold (float): Threshold for mean_std classification.

    Returns:
        ndarray: Enhanced image.
    """
    low_contrast, mean_diff, mean_std = is_low_contrast_clahe(
        img,
        lower_threshold=lower_contrast_threshold,
        upper_threshold=upper_contrast_threshold
    )

    low_contrast = (low_contrast and img[..., 1].max() == max_green_channel_value) \
        if mean_diff < mean_std_threshold else low_contrast

    if low_contrast:
        clip_limit = clip_limit_default
        kernel_size = kernel_size_default
        gamma = gamma_default

        if mean_diff > lower_contrast_threshold and mean_std < mean_std_threshold:
            clip_limit = clip_limit_high_diff
            kernel_size = kernel_size_high_diff
            gamma = gamma_high_diff
            model.bbox_threshold = bbox_threshold_high_diff

        if mean_diff > mean_diff_threshold and mean_std < mean_std_threshold:
            clip_limit = clip_limit_very_high_diff
            model.bbox_threshold = bbox_threshold_very_high_diff

        if mean_diff > mean_diff_threshold and (std_range[0] < mean_std < std_range[1]):
            clip_limit = clip_limit_adjusted

        img = equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit)
        img = adjust_gamma(img, gamma=gamma)

    return img


def get_median_size(labels):
    sizes = []
    sizes_abs = []
    for mask in np.unique(labels):
        if mask == 0:
            continue
        area = (labels == mask).sum().item()
        # normalizing by area
        sizes.append(area / (labels.shape[0] * labels.shape[1]))
        sizes_abs.append(area)
    sizes = np.array(sizes)
    sizes_abs = np.array(sizes_abs)
    # median size
    median_size = np.median(sizes)
    return median_size, sizes, sizes_abs


def is_low_contrast_clahe(image, lower_threshold=0.04, upper_threshold=0.05, kernel_size=256):
    cp = equalize_adapthist(image, kernel_size=kernel_size)
    diff = np.abs(image - cp)
    diff = diff[diff > 0]
    mean_diff = np.median(diff)
    mean_std = np.std(diff)
    print(f"Mean diff: {mean_diff}")
    print(np.mean(cp))
    islowcontrast = lower_threshold < mean_diff < upper_threshold
    return [islowcontrast, mean_diff, mean_std]


def add_white_borders(img, mask, color=None):
    if color is None:
        color = [1.0, 1.0, 1.0]
    assert img.shape[:2] == mask.shape
    assert img.shape[2] == 3
    assert len(img.shape) == 3

    boundary = _mask_outline(mask)
    img = np.array(img)  # copy
    r, c = np.where(np.isclose(1.0, boundary))
    img[r, c] = color
    return img

def subtract_boundaries(mask):
    boundaries = _mask_outline(mask)
    return mask - mask * boundaries


def _mask_outline(mask):
    """Create one pixel outline around maks."""
    outline = np.zeros_like(mask, dtype=np.uint8)
    outline[:, 1:][mask[:, :-1] != mask[:, 1:]] = 1
    outline[:-1, :][mask[:-1, :] != mask[1:, :]] = 1
    return outline

def _remap_array(mask, k, v):
    # k, v = np.array(list(mapper.keys())), np.array(list(mapper.values()))
    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype) #k,v from approach #1
    
    mapping_ar[k] = v
    return mapping_ar[mask]

def f1_score(pred, target):
    """ adapted from deepcell """

    pred_sum = np.count_nonzero(pred)
    target_sum = np.count_nonzero(target)

    # Calculations for IOU
    intersection = np.count_nonzero(np.logical_and(pred, target))
    # union = np.count_nonzero(np.logical_or(pred, target))
    recall = intersection / target_sum 
    precision = intersection /pred_sum 
    return hmean([recall, precision])


def relabel_mask(mask):
    """ Relabels the masks so indices are ordered by their position from the (0,0) point.
    mask (np.array): Input mask with two dimensions
    """
    assert mask.ndim == 2
    mask = relabel_sequential(np.array(mask))[0] # make sure it isn't a dask array
    n_cells = len(np.unique(mask)) - 1
    
    props = regionprops(mask)
    distances = []
    # remember one indexed
    
    for prop in props:
        r, c = prop.centroid
        distances.append(r**2 + c**2)
        
    remap = np.concatenate([[0], 1 + np.argsort(distances)])
    original_idxs = np.arange(n_cells + 1)
    
    assert n_cells + 1 == len(remap) == len(original_idxs)
    
    new_mask = _remap_array(mask, remap, original_idxs)
    assert f1_score(mask, new_mask) == 1.
    return new_mask

def format_image_shape(img):
    if len(img.shape) == 2:
        img = img[..., None]
    elif len(img.shape) == 3:
        channel_idx = np.argmin(img.shape)
        img = np.transpose(
            img, [i for i in range(len(img.shape)) if i != channel_idx] + [channel_idx]
        )
    else:
        raise ValueError(
            "Invalid image shape. Only (H, W) or (H, W, C) images are supported."
        )
    H, W, C = img.shape
    if C > 3:
        raise ValueError("A maximum of three channels is supported.")

    out_img = np.zeros((H, W, 3))
    out_img[:, :, -C:] = img
    return out_img


def normalize_image(image):
    image = format_image_shape(image)
    initial_shape = image.shape
    image = _histogram_normalization(_percentile_threshold(image))
    return image.reshape(initial_shape)


def _histogram_normalization(image, kernel_size=128):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.
    Args:
        image (numpy.array): Numpy array with shape (H, W, C).
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """

    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    for channel in range(image.shape[-1]):
        X = image[:, :, channel]
        sample_value = X[(0,) * X.ndim]
        if (X == sample_value).all():
            image[:, :, channel] = np.zeros_like(X)
            continue

        X = rescale_intensity(X, out_range=(0.0, 1.0))
        X = equalize_adapthist(X, kernel_size=kernel_size)
        image[:, :, channel] = X

    return image


def _percentile_threshold(image, percentile=99.9):
    """Threshold an image to reduce bright spots
    Args:
        image: numpy array of image data. Should have shape (H, W, C)
        percentile: cutoff used to threshold image
    Returns:
        np.array: thresholded version of input image
    """
    processed_image = np.zeros_like(image)
    for chan in range(image.shape[-1]):
        current_img = np.copy(image[..., chan])
        non_zero_vals = current_img[np.nonzero(current_img)]

        # only threshold if channel isn't blank
        if len(non_zero_vals) > 0:
            img_max = np.percentile(non_zero_vals, percentile)

            # threshold values down to max
            threshold_mask = current_img > img_max
            current_img[threshold_mask] = img_max

            # update image
            processed_image[..., chan] = current_img

    return processed_image


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """
    From https://github.com/MouseLand/cellpose/blob/509ffca33737058b0b4e2e96d506514e10620eb3/cellpose/utils.py#L616

    Fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    (might have issues at borders between cells, todo: check and fix)

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim
        )

    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:
                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = j + 1
                j += 1
    return masks
