import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, find_objects
from skimage.exposure import equalize_adapthist, rescale_intensity

def normalize_image(image):
    return _histogram_normalization(_percentile_threshold(image))


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

    assert image.dtype == np.float32
    assert len(image.shape) == 3

    for channel in range(image.shape[-1]):
        X = image[:, :, channel]
        sample_value = X[(0,) * X.ndim]
        if (X == sample_value).all():
            # TODO: Deal with constant value arrays
            # https://github.com/scikit-image/scikit-image/issues/4596
            image[:, :, channel] = np.zeros_like(X)
            continue

        X = rescale_intensity(X, out_range=(0.0, 1.0))
        X = equalize_adapthist(X, kernel_size=kernel_size)
        image[:, :, channel] = X
    return image


def _percentile_threshold(image, percentile=99.9):
    """Threshold an image to reduce bright spots
    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image
    Returns:
        np.array: thresholded version of input image
    """
    # image is B, H, W, C
    processed_image = np.zeros_like(image)
    for img in range(image.shape[0]):
        for chan in range(image.shape[-1]):
            current_img = np.copy(image[img, ..., chan])
            non_zero_vals = current_img[np.nonzero(current_img)]

            # only threshold if channel isn't blank
            if len(non_zero_vals) > 0:
                img_max = np.percentile(non_zero_vals, percentile)

                # threshold values down to max
                threshold_mask = current_img > img_max
                current_img[threshold_mask] = img_max

                # update image
                processed_image[img, ..., chan] = current_img

    return processed_image


def keep_largest_object(img):
    """
    Keep only the largest object in the binary image (np.array).
    """
    img_array = img

    label_image, num_features = ndimage.label(img_array)

    label_histogram = np.bincount(label_image.ravel())
    label_histogram[0] = 0  # Clear the background label

    largest_object_label = label_histogram.argmax()
    cleaned_array = np.where(label_image == largest_object_label, img_array.max(), 0)

    return cleaned_array


### from cellpose repo
def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

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
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

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
                masks[slc][msk] = (j + 1)
                j += 1
    return masks
