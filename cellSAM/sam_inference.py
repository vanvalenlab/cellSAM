from typing import Tuple
import copy

import numpy as np
import warnings
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
from torchvision import tv_tensors
from sklearn.cluster import KMeans

import torchvision.transforms.v2 as T
from .AnchorDETR.models import build_inference_model
from .AnchorDETR import transforms as anchorT
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
        
        # Set all parameters from modelconfig.yaml
        args.enc_layers = 6
        args.dec_layers = 6
        args.dim_feedforward = 1024
        args.hidden_dim = 256
        args.dropout = 0.0
        args.nheads = 8
        args.num_query_position = 3500
        args.num_query_pattern = 1
        args.spatial_prior = "learned"
        args.attention_type = "RCDA"
        args.num_feature_levels = 1
        args.device = "cuda"
        args.num_classes = 2
        
        # Additional required parameters
        args.in_channels = 768
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
        self.bbox_threshold = 0.4
        self.sam_transform = ResizeLongestSide(1024)

        config = Namespace(**config)
        self.cellfinder = CellfinderAnchorDetr(config)

        self.adv_mode = True
        self.model_cp = copy.deepcopy(self.model)

        # Transforms
        self.normalize = T.Compose([
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_transforms(self, imgs):
        imgs = [tv_tensors.Image(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)

        return imgs

    def sam_preprocess(self, x: torch.Tensor, return_paddings=False, div_255=False):
        """Normalize pixel values and pad to a square input."""
        mean = self.model.pixel_mean
        std = self.model.pixel_std
        if div_255:
            mean = mean / 255
            std = std / 255
        x = (x - mean) / std

        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if return_paddings:
            return x, (padh, padw)
        else:
            return x

    def sam_bbox_preprocessing(self, imgs, percentile=True):
        imgs = [T.Resize((1024, 1024))(img) for img in imgs]
        x = [torchvision.transforms.ToPILImage()(img) for img in imgs]
        x = [np.array(img) for img in x]
        x = [self.sam_transform.apply_image(img) for img in x]
        device = next(self.parameters()).device
        x = [torch.from_numpy(img).permute(2, 0, 1).contiguous().to(device) for img in x]
        x = [self.sam_preprocess(img, return_paddings=True) for img in x]
        imgs, paddings = zip(*x)
        if percentile:
            imgs = [anchorT.PercentileThreshold()(img.cpu()) for img in imgs]
        imgs = [torch.Tensor(img) for img in imgs]
        imgs = [self.normalize(img) for img in imgs]
        imgs = [anchorT.Standardize()(img) for img in imgs]

        if self.adv_mode:
            imgs = [anchorT.ToRGB()(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        device = next(self.parameters()).device
        imgs = imgs.to(device)

        return imgs

    def sam_preprocess_pad(self, x: torch.Tensor, return_paddings=False):
        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if return_paddings:
            return x, (padh, padw)
        else:
            return x

    def forward(self, x, return_preprocessed=False, *args, **kwargs):
        device = next(self.parameters()).device
        x = x.to(device)
        x = [self.sam_preprocess(img, return_paddings=True, div_255=True) for img in x]
        x, paddings = zip(*x)
        preprocessed_img = torch.stack(x, dim=0)

        if self.adv_mode:
            x = self.model_cp.image_encoder(preprocessed_img)
        else:
            x = self.model.image_encoder(preprocessed_img)

        if return_preprocessed:
            return x, preprocessed_img, paddings
        else:
            return x

    def prep_2(self, imgs, percentile=True):
        imgs = [T.Resize((1024, 1024))(img) for img in imgs]
        imgs = [self.sam_preprocess_pad(img, return_paddings=True) for img in imgs]
        imgs, paddings = zip(*imgs)

        if percentile:
            imgs = [anchorT.PercentileThreshold()(img.cpu()) for img in imgs]
        imgs = [torch.Tensor(img) for img in imgs]
        imgs = [self.normalize(img) for img in imgs]
        imgs = [anchorT.Standardize()(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)

        return imgs, paddings

    @torch.no_grad()
    def generate_bounding_boxes(self, images, device=None):
        """
        Generates bounding boxes for the given images with dynamic thresholding.
        """
        transformed_imgs_anchor = self.sam_bbox_preprocessing(images, percentile=not self.adv_mode)
        results = self.cellfinder.forward_inference(transformed_imgs_anchor)

        boxes_per_heatmap = [x["boxes"] for x in results]
        pred_scores = [x["scores"] for x in results]

        # Apply dynamic thresholding like in predict method
        if len(pred_scores) > 0:
            data = pred_scores[0].detach().cpu().numpy()
            data_reshaped = data.reshape(-1, 1)

            # dynamic thresholding
            threshold = self.bbox_threshold
            if len(data_reshaped) > 1:
                try:
                    kmeans = KMeans(n_clusters=2, random_state=42).fit(data_reshaped)
                    cluster_centers = kmeans.cluster_centers_
                    threshold_cluster = np.mean(cluster_centers)
                    threshold = 0.66 * self.bbox_threshold + 0.33 * threshold_cluster
                except:
                    pass

            if threshold > 0.0 and len(data) > 0:
                boxes_per_heatmap = [
                    box[data > threshold] for idx, box in enumerate(boxes_per_heatmap)
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
            transformed_imgs_anchor, paddings = self.prep_2(images, percentile=True)
            x = self.forward(transformed_imgs_anchor, return_preprocessed=False)
            return x, paddings
        else:
            # Use existing embeddings and compute paddings
            paddings = []
            for img in images:
                h, w = img.shape[-2:]
                paddings.append((1024 - h, 1024 - w))
            return existing_embeddings, paddings

    def predict(self, images, coords_per_heatmap=None, boxes_per_heatmap=None):
        device = next(self.parameters()).device

        assert self.mask_threshold > 0

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        x, paddings = self.generate_embeddings(images, device=device)

        if coords_per_heatmap is None and boxes_per_heatmap is None:
            boxes_per_heatmap = self.generate_bounding_boxes(images, device=device)

        for idx in range(len(x)):
            boxes = boxes_per_heatmap[idx] if idx < len(boxes_per_heatmap) else boxes_per_heatmap[0]
            rng = len(boxes)
            low_masks = []
            low_masks_thresholded = []
            scores = []
            final_boxes = []
            mdl = self.model_cp if self.adv_mode else self.model

            for coord_idx in range(rng):
                bbox = boxes[coord_idx]
                input_box = torch.as_tensor(bbox).unsqueeze(0).unsqueeze(0)

                sparse_embeddings, dense_embeddings = mdl.prompt_encoder(
                    points=None,
                    boxes=input_box.to(device),
                    masks=None,
                )

                low_res_masks, iou_predictions = mdl.mask_decoder(
                    image_embeddings=x[idx].unsqueeze(0).to(device),
                    image_pe=mdl.prompt_encoder.get_dense_pe(),
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
                    input_size=torch.tensor([1024 - paddings[idx][0], 1024 - paddings[idx][1]]).to(device),
                    original_size=images[idx].shape[-2:]
                )
                low_res_masks_thresholded = nn.Sigmoid()(low_res_masks[0, 0]) > self.mask_threshold
                low_res_masks_thresholded = low_res_masks_thresholded.numpy().astype(np.uint8)

                res = low_res_masks[0, 0].detach().cpu().numpy()

                low_masks.append(res)
                low_res_masks_thresholded = low_res_masks_thresholded[:images[idx].shape[1], :images[idx].shape[2]]
                low_masks_thresholded.append(low_res_masks_thresholded)
                scores.append(float(iou_predictions[0][0].detach().cpu().numpy()))

                # Scale bbox back to original image size
                _bbox = [b.cpu().numpy() if hasattr(b, 'cpu') else b for b in bbox]
                im_w = images[0].shape[2]
                im_h = images[0].shape[1]
                scale_x = im_w / 1024
                scale_y = im_h / 1024
                _bbox = [
                    _bbox[0] * scale_x,
                    _bbox[1] * scale_y,
                    _bbox[2] * scale_x,
                    _bbox[3] * scale_y,
                ]
                final_boxes.append(_bbox)

            if low_masks:
                thresholded_masks = np.stack(low_masks_thresholded)
                final_boxes = np.stack(final_boxes)

                # Create instance segmentation mask
                thresholded_masks_summed = (
                        thresholded_masks * np.arange(1, thresholded_masks.shape[0] + 1)[:, None, None]
                )
                thresholded_masks_summed = np.max(thresholded_masks_summed, axis=0)

                return thresholded_masks_summed, thresholded_masks, x, final_boxes
            else:
                return None, None, None, None

    def load_state_dict(self, state_dict, strict=True):

        if isinstance(state_dict, dict) and 'state_dict' in state_dict and not any(
                k.startswith('model.') for k in state_dict.keys()):
            state_dict = state_dict['state_dict']
        has_model_cp = any(k.startswith('model_cp.') for k in state_dict.keys())
        result = super().load_state_dict(state_dict, strict=strict)
        if not has_model_cp:
            self.adv_mode = False
            self.model_cp.load_state_dict(self.model.state_dict(), strict=False)
        return result
