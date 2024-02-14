import torch
import numpy as np
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.morphology import disk, binary_opening, binary_closing, binary_erosion, binary_dilation
from scipy.ndimage import gaussian_filter
from segment_anything.utils.amg import remove_small_regions


from utils import keep_largest_object, fill_holes_and_remove_small_masks

def image_casework(img):
    if len(img.shape) == 2:
        img = img[..., None]
    elif len(img.shape) == 3:
        channel_idx = np.argmin(img.shape)
        img = np.transpose(img, [i for i in range(len(img.shape)) if i != channel_idx] + [channel_idx])
    else:
        raise ValueError("Invalid image shape! Image must be H, W, C")
    H, W, C = img.shape
    out_img = np.zeros((H, W, 3))
    out_img[:, :, -C:] = img
    return out_img

def process_and_segment_image(model, img, normalize=False, postprocess=False):
    """
    model (nn.Module): Loaded CellSAM model
    img  (np.array): Image to be segmented with shape (H, W, C)
    """
    img = image_casework(img)
    H, W, C = img.shape
    if C > H or C > W:
        raise ValueError("Number of channels is greater than H or W. ")

    if normalize:
        img = normalize_image(img.astype(np.float32))

    img = img.transpose((2,0,1)) # channel first for pytorch.
    img = torch.from_numpy(img).float().cuda().unsqueeze(0)

    segmentation_predictions, thresholded_masks, low_masks, scores, x, bounding_boxes = model.predict(
        img,
        prompts=["box"],
        x=None,
        return_lower_level_comps=True,
        boxes_per_heatmap=None,
    )


    for mask_idx, msk in enumerate(thresholded_masks):
        msk = keep_largest_object(msk)
        thresholded_masks[mask_idx] = msk

    thresholded_masks_summed = (
        thresholded_masks * np.arange(1, thresholded_masks.shape[0] + 1)[:, None, None]
    )
    # sum all masks, #TODO: double check if max is the right move here
    segmentation_predictions = np.max(thresholded_masks_summed, axis=0)

    if postprocess:
        mask_values  = np.unique(segmentation_predictions)
        new_masks = []
        for mask_value in mask_values[1:]:
            mask = segmentation_predictions == mask_value
            mask, changed = remove_small_regions(mask, 20, mode="holes")
            mask, changed = remove_small_regions(mask, 20, mode="islands")
            selem = disk(2)
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

        segmentation_predictions = np.max(new_masks, axis=0)

    return (
        fill_holes_and_remove_small_masks(segmentation_predictions, min_size=25),
        x.cpu().numpy(),
    )

def normalize_image(image):
    return histogram_normalization(percentile_threshold(image))

def histogram_normalization(image, kernel_size=128):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.
    Args:
        image (numpy.array): numpy array of phase image data.
            B, H, W, C
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """

    assert image.dtype == np.float32
    assert len(image.shape) == 3

    for channel in range(image.shape[-1]):
        X = image[..., channel]
        sample_value = X[(0,) * X.ndim]
        if (X == sample_value).all():
            # TODO: Deal with constant value arrays
            # https://github.com/scikit-image/scikit-image/issues/4596
            image[..., channel] = np.zeros_like(X)
            continue

        X = rescale_intensity(X, out_range=(0.0, 1.0))
        X = equalize_adapthist(X, kernel_size=kernel_size)
        image[..., channel] = X
    return image


def percentile_threshold(image, percentile=99.9):
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

if __name__ == '__main__':
    # model stuff
    model_weight_path = "/home/rdilip/.cellsam_assets/he_inference_model.pth"
    model = torch.load(model_weight_path, map_location=torch.device("cpu"))
    model.bbox_threshold = 0.4
    model.mask_threshold = 0.4
    model.iou_threshold = 0.5
    # Initialize
    model.eval()
    model.cuda()