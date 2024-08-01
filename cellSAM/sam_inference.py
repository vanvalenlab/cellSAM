from typing import Tuple

import numpy as np
import warnings
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
from torchvision import datapoints

from .AnchorDETR.models import build_inference_model
from segment_anything import (
    sam_model_registry,
)
from scipy import ndimage


def keep_largest_object(img: np.ndarray) -> np.ndarray:
    """
    Keep only the largest object in the binary image (np.array).
    """
    img_array = img
    label_image, _ = ndimage.label(img_array)
    label_histogram = np.bincount(label_image.ravel())
    label_histogram[0] = 0  # Clear the background label
    largest_object_label = label_histogram.argmax()
    cleaned_array = np.where(label_image == largest_object_label, img_array.max(), 0)

    return cleaned_array


# from sam repo
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CellfinderAnchorDetr(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        args.num_classes = 2
        args.in_channels = 768
        args.device = "cuda"
        args.backbone = "SAM"
        args.only_neck = False
        args.freeze_backbone = False
        args.sam_vit = "vit_b"

        if not hasattr(self, "decode_head"):
            self.decode_head, self.postprocessors = build_inference_model(args)

    def forward(self, features=None):
        outputs = self.decode_head(features)
        return outputs

    @torch.no_grad()
    def forward_inference(self, imgs, viz=False):
        outputs = self.decode_head(imgs)

        orig_target_sizes = [torch.tensor(img.shape[-2:]) for img in imgs]
        orig_target_sizes = torch.stack(orig_target_sizes, dim=0)
        orig_target_sizes = orig_target_sizes.to(imgs.device)

        res = self.postprocessors["bbox"](outputs, orig_target_sizes)
        return res


class CellSAM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = sam_model_registry["vit_b"]()
        self.mask_threshold = 0.4
        self.iou_threshold = 0.5
        self.bbox_threshold = 0.2
        self.sam_transform = ResizeLongestSide(1024)

        config = Namespace(**config)
        self.cellfinder = CellfinderAnchorDetr(config)

    def predict_transforms(self, imgs):
        imgs = [datapoints.Image(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)

        return imgs

    def sam_preprocess(self, x: torch.Tensor, return_paddings=False):
        """Normalize pixel values and pad to a square input."""
        x = (x - self.model.pixel_mean) / self.model.pixel_std

        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if return_paddings:
            return x, (padh, padw)
        else:
            return x

    def sam_bbox_preprocessing(self, x, device=None):
        x = [torchvision.transforms.ToPILImage()(img) for img in x]
        x = [np.array(img) for img in x]
        x = [self.sam_transform.apply_image(img) for img in x]
        x = [
            torch.from_numpy(img).permute(2, 0, 1).contiguous().to(device) for img in x
        ]
        x = [self.sam_preprocess(img, return_paddings=True) for img in x]
        x, paddings = zip(*x)
        preprocessed_img = torch.stack(x, dim=0)
        return preprocessed_img, paddings

    def forward(self, x, return_preprocessed=False, device=None):
        preprocessed_img, paddings = self.sam_bbox_preprocessing(x, device=device)
        x = self.model.image_encoder(preprocessed_img)

        if return_preprocessed:
            return x, preprocessed_img, paddings
        else:
            return x

    @torch.no_grad()
    def generate_bounding_boxes(self, images, device=None):
        """
        Generates bounding boxes for the given images.
        """
        processed_imgs, _ = self.sam_bbox_preprocessing(images, device=device)
        results = self.cellfinder.forward_inference(processed_imgs)

        boxes_per_heatmap = [x["boxes"] for x in results]
        pred_scores = [x["scores"] for x in results]

        if self.bbox_threshold > 0.0:
            boxes_per_heatmap = [
                box[pred_scores[idx] > self.bbox_threshold]
                for idx, box in enumerate(boxes_per_heatmap)
            ]

        return boxes_per_heatmap

    @torch.no_grad()
    def generate_embeddings(
        self, images, existing_embeddings=None, transform=True, device=None
    ):
        """
        Generates embeddings for the given images or uses existing embeddings if provided.
        """
        if existing_embeddings is None:
            transformed_images = (
                self.predict_transforms(images) if transform else images
            )
            embeddings, _, paddings = self(
                transformed_images, return_preprocessed=True, device=device
            )
        else:
            embeddings = existing_embeddings
            # Compute paddings for existing embeddings
            paddings = []
            IMG_SIZE = self.model.image_encoder.img_size
            for img in images:
                h, w = img.shape[-2:]
                scale = int(1024 / max(h, w))
                h, w = h * scale, w * scale
                paddings.append((IMG_SIZE - h, IMG_SIZE - w))

        return embeddings, paddings

    @torch.no_grad()
    def predict(
        self, images, boxes_per_heatmap=None, transform=True, x=None, device=None, fast=False
    ):
        assert self.mask_threshold > 0  # otherwise all pred. will be true-> no blobs
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if x is None:
            x, paddings = self.generate_embeddings(
                images, transform=transform, device=device
            )
        else:
            paddings = []
            for img in images:
                h, w = img.shape[-2:]
                scaling = 1024 / max(h, w)
                paddings.append(
                    (1024 - int(h * scaling), 1024 - int(w * scaling))
                )

        # In principle, we can batch images, but I suspect it will cause problems
        # with simultaneously batching the bounding boxes, which is more useful for inference.
        assert x.size(0) == 1
                

        # TODO: update to use existing features
        if boxes_per_heatmap is None:
            boxes_per_heatmap = self.generate_bounding_boxes(images, device=device)
        else:
            boxes_per_heatmap = (
                torch.from_numpy(np.array(boxes_per_heatmap) * 1024 / max(images[0].shape))
            )
        
        # B, N, 4
        if not fast:
            boxes_per_heatmap = boxes_per_heatmap[0]


        low_masks = []
        low_masks_thresholded = []
        scores = []


        for input_bbox in boxes_per_heatmap:
            # if fast , passes N, 4
            # else, passes 4
            while len(input_bbox.shape) < 2:
                input_bbox = input_bbox.unsqueeze(0)
            input_bbox = input_bbox.to(device)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=input_bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=x.to(device),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            low_res_masks = low_res_masks.detach().cpu()

            # threshold based on iou predictions
            if iou_predictions[0][0] < self.iou_threshold:
                warnings.warn("Low IOU threshold, ignoring mask.")
                continue

            low_res_masks = self.model.postprocess_masks(
                low_res_masks,
                input_size=torch.tensor(
                    [1024 - paddings[0][0], 1024 - paddings[0][1]]
                ).to(device),
                original_size=images.shape[-2:],
            )

            low_res_masks_thresholded = (
                nn.Sigmoid()(low_res_masks[:, 0]) > self.mask_threshold
            )

            low_res_masks_thresholded = low_res_masks_thresholded.numpy().astype(
                np.uint8
            )

            res = low_res_masks[:, 0].detach().cpu().numpy()

            assert res.shape[-2:] == images[0].shape[1:]
            low_masks.append(res)
            low_res_masks_thresholded = low_res_masks_thresholded[ :,
                : images[0].shape[1], : images[0].shape[2]
            ]
            low_masks_thresholded.append(low_res_masks_thresholded)
            scores.append(iou_predictions[:, 0].detach().cpu().numpy())


        if low_masks == []:
            return None, None, None, None
        

        if fast:
            low_masks = low_masks[0]
            thresholded_masks = low_masks_thresholded[0]
            scores = np.array(scores)
        else:
            low_masks = np.stack(low_masks)
            thresholded_masks = np.stack(low_masks_thresholded)
            scores = np.stack(scores)

        low_masks, thresholded_masks, scores = map(
            np.squeeze,
            (low_masks, thresholded_masks, scores)
        )
        

        for mask_idx, msk in enumerate(thresholded_masks):
            thresholded_masks[mask_idx] = keep_largest_object(msk)

        # multiply each mask by its index
        thresholded_masks_summed = (
            thresholded_masks
            * np.arange(1, thresholded_masks.shape[0] + 1)[:, None, None]
        )

        # sum all masks, #TODO: double check if max is the right move here
        thresholded_masks_summed = np.max(thresholded_masks_summed, axis=0)

        return thresholded_masks_summed, thresholded_masks, x, boxes_per_heatmap
