import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, find_objects
from skimage.exposure import equalize_adapthist, rescale_intensity


def subtract_boundaries(mask):
    boundaries = _mask_outline(mask)
    return mask - mask * boundaries


def _mask_outline(mask):
    """Create one pixel outline around maks."""
    outline = np.zeros_like(mask, dtype=np.uint8)
    outline[:, 1:][mask[:, :-1] != mask[:, 1:]] = 1
    outline[:-1, :][mask[:-1, :] != mask[1:, :]] = 1
    return outline


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
    initial_shape = image.shape
    image = format_image_shape(image)
    image = _histogram_normalization(_percentile_threshold(image))
    return image.reshape(initial_shape)


def _histogram_normalization(image, kernel_size=128):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.
    Args:
        image (numpy.array): numpy array of phase image data.
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
