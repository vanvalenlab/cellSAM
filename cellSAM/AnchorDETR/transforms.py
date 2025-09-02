# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Transforms and data augmentation for both image + bbox.
"""
import math
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageDraw
from skimage.exposure import equalize_adapthist, rescale_intensity

from cellSAM.AnchorDETR.util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from cellSAM.AnchorDETR.util.misc import interpolate


def visualize_boxes(image, boxes, color="red", width=2):
    """
    Visualize bounding boxes on a PIL image.

    Args:
    - image (PIL.Image): The image on which to draw the bounding boxes.
    - boxes (list or tensor): List of bounding boxes in [x1, y1, x2, y2] format.
    - color (str or tuple): Color for the bounding boxes.
    - width (int): Line width of the bounding boxes.

    Returns:
    - PIL.Image: Image with drawn bounding boxes.
    """

    # Convert image to RGB mode (for drawing)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        draw.rectangle(
            [x_min, y_min, x_max, y_max], outline=color, width=width
        )

    image.show()
    return image


class RandomRotate(object):
    def __init__(self, angles):
        assert isinstance(angles, (list, tuple))
        self.angles = angles

    def __call__(self, img, target=None):
        angle = random.choice(self.angles)
        return rotate(img, target, angle)


def rotate(image, target, angle):
    rotated_image = F.rotate(image, angle)

    target = target.copy()
    w, h = image.size
    cx, cy = w / 2, h / 2
    rad_angle = math.radians(angle)  # Changed here to adjust the rotation direction

    keep_indices = []

    if "boxes" in target:
        boxes = target["boxes"]
        new_boxes = []
        for idx, box in enumerate(boxes):
            corners = torch.tensor([
                [box[0], box[1]],  # Top-left corner
                [box[2], box[1]],  # Top-right corner
                [box[2], box[3]],  # Bottom-right corner
                [box[0], box[3]],  # Bottom-left corner
            ])

            # Translate corners to origin based on center of image
            corners -= torch.tensor([cx, cy])

            # Rotate corners
            rotation_matrix = torch.tensor([
                [math.cos(rad_angle), -math.sin(rad_angle)],
                [math.sin(rad_angle), math.cos(rad_angle)]
            ], dtype=torch.float32)
            rotated_corners = torch.matmul(corners, rotation_matrix)

            # Translate corners back
            rotated_corners += torch.tensor([cx, cy])

            # Get the min and max points
            min_xy = torch.clamp(torch.min(rotated_corners, dim=0)[0], 0, max(w - 1, h - 1))  # Clamp to image size
            max_xy = torch.clamp(torch.max(rotated_corners, dim=0)[0], 0, max(w - 1, h - 1))  # Clamp to image size

            # Create a new box
            new_box = torch.cat((min_xy, max_xy)).unsqueeze(0)

            new_boxes.append(new_box)
            keep_indices.append(idx)

        # Filter other target fields based on keep_indices
        if keep_indices:
            target["boxes"] = torch.cat(new_boxes, dim=0)
            target["area"] = target["area"][keep_indices]
            if "labels" in target:
                target["labels"] = target["labels"][keep_indices]
            if "iscrowd" in target:
                target["iscrowd"] = target["iscrowd"][keep_indices]
            if "masks" in target:
                target["masks"] = target["masks"][keep_indices]
        else:
            target["boxes"] = torch.empty((0, 4))
            target["area"] = torch.empty((0,))
            if "labels" in target:
                target["labels"] = torch.empty((0,))
            if "iscrowd" in target:
                target["iscrowd"] = torch.empty((0,))
            if "masks" in target:
                target["masks"] = torch.empty((0, h, w))

    return rotated_image, target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def vflip(image, target):
    flipped_image = F.vflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [0, 3, 2, 1]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-2)  # flip along the vertical axis

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    w, h = padded_image.size
    target["size"] = torch.tensor([h, w])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class FixedResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = self.sizes  # directly use the given size
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSizeCrop_same(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        crop_size = random.randint(self.min_size, min(img.width, img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [crop_size, crop_size])
        return crop(img, target, region)


class RandomPad_same(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad = random.randint(0, self.max_pad)
        return pad(img, target, (pad, pad))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


import torchvision.transforms.v2 as T2


# random color augmentations
class RandomColorAugmentation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img)

        return img, target


class RandomClahe(object):
    def __init__(self, clip_limit=4.0, tile_grid_size=(128, 128)):
        from kornia.enhance import equalize_clahe


        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img, target):
        img = equalize_clahe(img.unsqueeze(0), clip_limit=random.uniform(1.5, self.clip_limit),
                             grid_size=self.tile_grid_size)
        return img, target


class Clahe(object):
    def __init__(self, clip_limit=4.0, tile_grid_size=(256, 256)):
        from kornia.enhance import equalize_clahe


        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img, backend="kornia"):
        if backend == "kornia":
            img = equalize_clahe(img.unsqueeze(0), clip_limit=self.clip_limit, grid_size=self.tile_grid_size)
        else:
            img = rescale_intensity(img.cpu().numpy(), out_range=(0.0, 1.0))
            img = equalize_adapthist(img, kernel_size=128)
        img = equalize_clahe(img.unsqueeze(0), clip_limit=1.0,
                             grid_size=self.tile_grid_size)
        return img


class Standardize(object):
    def __init__(self):
        from kornia.enhance import normalize_min_max

    def __call__(self, image, target=None):
        image = image.unsqueeze(0)
        image = normalize_min_max(image)
        image = image.squeeze(0)
        if target is None:
            return image
        return image, target


class mediar(object):
    def __call__(self, image, target):
        image = image.median_filter(3)
        return image, target


from skimage import exposure
import numpy as np
from PIL import Image


class PercentileThreshold(object):
    def __init__(self, lower=0.0, upper=99.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, img, tgt=None):
        # is PIL?
        is_pil = isinstance(img, Image.Image)

        img = np.array(img)
        non_zero_vals = img[np.nonzero(img)]
        if len(non_zero_vals) == 0:
            return img, tgt
        percentiles = np.percentile(non_zero_vals, [self.lower, self.upper])
        img = exposure.rescale_intensity(
            img, in_range=(percentiles[0], percentiles[1]), out_range="float32"
        )
        if is_pil:
            img = img.astype(np.uint8)
            # back to PIL
            img = Image.fromarray(img)

        if tgt is None:
            return img
        return img, tgt


class PercentileThreshold_a(object):
    def __init__(self, lower=0.0, upper=99.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, img, tgt=None):
        # is PIL?
        is_pil = isinstance(img, Image.Image)

        img = np.array(img)
        non_zero_vals = img[np.nonzero(img)]
        if len(non_zero_vals) == 0:
            return img, tgt
        percentiles = np.percentile(non_zero_vals, [self.lower, self.upper])
        img = exposure.rescale_intensity(
            img, in_range=(percentiles[0], percentiles[1]), out_range="uint8"
        )
        # if is_pil:
        #     img = img.astype(np.uint8)
        #     back to PIL
        # img = Image.fromarray(img)

        if tgt is None:
            return img
        return img, tgt


import torchvision.transforms.v2 as tv_transforms


class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            img = tv_transforms.GaussianNoise(mean=self.mean, sigma=self.std)(img)
        return img, tgt


class RandomGamma(object):
    def __init__(self, gamma=(0.5, 1.5), p=0.5):
        from kornia.enhance import adjust_gamma


        self.gamma = gamma
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            img = adjust_gamma(img, gamma=random.uniform(*self.gamma))
        return img, tgt


class RandomGauss(object):
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            img = tv_transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))(img)
        return img, tgt


class RandomSharpening(object):
    def __init__(self, alpha=(0.5, 1.0), p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            img = tv_transforms.RandomAdjustSharpness(sharpness_factor=random.uniform(*self.alpha))(img)
        return img, tgt


class RandomEqualize(object):
    def __call__(self, img, tgt, p=0.5):
        if random.random() < p:
            img = tv_transforms.RandomEqualize(p=1.0)(img)
        return img, tgt


class ToRGB(object):
    def __call__(self, img, tgt=None):
        if torch.max(img[1, ...]) == 0:
            img[1, ...] = img[2, ...]
        if torch.max(img[2, ...]) == 0:
            img[2, ...] = img[1, ...]
        if tgt is None:
            return img
        return img, tgt


class SwapChannels(object):
    def __call__(self, img, tgt):
        # C H W to H W C
        img = img.permute(1, 2, 0)
        return img, tgt


class RandomHistogram(object):
    def __init__(self, num_control_points=3, p=0.5):
        from monai.transforms import RandHistogramShift


        self.num_control_points = num_control_points
        self.p = p
        self.aug = RandHistogramShift(num_control_points=num_control_points, prob=1.0)

    def __call__(self, img, tgt):
        if random.random() < self.p:
            img = self.aug(img)
        return img, tgt


class RandomZoomOut(object):
    def __init__(self, max_zoom_out=0.5, p=0.5):
        self.max_zoom_out = max_zoom_out
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < 1.0:
            # Apply random zoom-out transformation to the image
            zoom_out = random.uniform(1.2, self.max_zoom_out)
            img = tv_transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(zoom_out, zoom_out), shear=0)(img)

            # Get image dimensions
            w, h = img.size

            # Compute the amount of padding added to the image
            # The padding is (total_padding_w, total_padding_h), so divide by 2 for padding on each side
            pad_w = (w - w * zoom_out) / 2
            pad_h = (h - h * zoom_out) / 2

            # Update the bounding boxes after zoom-out transformation
            boxes = tgt["boxes"]  # Boxes are in xyxy format

            # First, scale the bounding box coordinates relative to the zoom-out factor
            boxes[:, 0] = boxes[:, 0] * zoom_out + pad_w  # x_min
            boxes[:, 1] = boxes[:, 1] * zoom_out + pad_h  # y_min
            boxes[:, 2] = boxes[:, 2] * zoom_out + pad_w  # x_max
            boxes[:, 3] = boxes[:, 3] * zoom_out + pad_h  # y_max

            # Clamp the bounding boxes to make sure they stay within image bounds
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)  # Clamp x coordinates
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)  # Clamp y coordinates

            # Update the target with new bounding boxes
            tgt["boxes"] = boxes
        return img, tgt
