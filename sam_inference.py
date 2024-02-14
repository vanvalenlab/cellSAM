from copy import deepcopy
from typing import Tuple

import lightning
import numpy as np
from torchmetrics.detection import MeanAveragePrecision
import yaml
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
from torchvision import datapoints


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

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[2], image.shape[3], self.target_length
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=False
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

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
    def __init__(self, cfg, args, device=None):
        super().__init__()

        self.cfg = cfg
        self.args = args
        # pascal classes
        if self.cfg["data"]["name"] == "pascal":
            args.num_classes = 21
        else:
            args.num_classes = 2
        args.in_channels = 768
        args.device = "cuda"
        args.backbone = "SAM"
        args.only_neck = False
        args.freeze_backbone = False
        args.sam_vit = "vit_b"
        if not hasattr(self, "decode_head"):
            from AnchorDETR.models import build_model

            self.decode_head, self.criterion, self.postprocessors = build_model(args)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, targets=None, features=None, device=None, **kwargs):
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


class SAM(lightning.pytorch.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if not hasattr(self, "model"):
            from segment_anything.segment_anything import (
                sam_model_registry,
            )

            self.model = sam_model_registry[cfg["sam"]["model_type"]](
                checkpoint=cfg["sam"]["backbone"]
            )
        self.criterion = nn.BCEWithLogitsLoss()
        self.coord_loss = nn.MSELoss()
        self.mask_threshold = 0.4
        self.iou_threshold = 0.5
        self.bbox_threshold = 0.2
        self.use_nms = 0
        self.number_of_points = 1
        self.img_size = self.cfg["data"]["img_size"]
        self.img_scale_factor = int(1024 / self.img_size)
        self.embedding_dimension = 256
        self.dense_point_embedding_size = (1024, 1024)
        self.sam_transform = ResizeLongestSide(1024)
        self.target_image_size = self.cfg["data"]["img_size"]
        self.cellfinder_threshold = 0.5
        self.img_size = 224

        self.sam_vit_in_features = 256
        self.sam_vit_feature_dim = 64

        for param in self.model.parameters():
            param.requires_grad = False
        anchor_cfg = yaml.load(open("anchordetr.yaml", "r"), Loader=yaml.FullLoader)
        anchor_cfg = Namespace(**anchor_cfg)
        anchor_cfg.num_query_position = self.cfg["num_query_position"]
        anchor_cfg.num_query_pattern = self.cfg["num_query_pattern"]
        anchor_cfg.spatial_prior = self.cfg["spatial_prior"]
        self.cellfinder = CellfinderAnchorDetr(self.cfg, anchor_cfg, self.device)
        self.cellfinder = self.cellfinder.to(self.device)

        self.sem_seg = self.cfg["sem_seg"]
        self.prompter = self.cfg["prompter"]
        self.max_count = self.cfg["max_count"]
        self.blurr_sigma = self.cfg["blurr_sigma"]

        # detection params
        self.max_sigma = 3
        self.num_sigma = 3
        self.blur_sigma = 1.0
        # self.threshold = 0.05
        self.threshold = 0.1
        self.detect_method = "log"
        self.global_count = 0
        # self.sem_seg = "pretrain"

    def forward(
        self,
        x,
        return_heatmaps=False,
        multimask_output=False,
        return_preprocessed=False,
        *args,
        **kwargs
    ):
        x = [torchvision.transforms.ToPILImage()(img) for img in x]
        x = [np.array(img) for img in x]
        x = [self.sam_transform.apply_image(img) for img in x]
        x = [
            torch.from_numpy(img).permute(2, 0, 1).contiguous().to(self.device)
            for img in x
        ]
        x = [self.sam_preprocess(img, return_paddings=True) for img in x]
        x, paddings = zip(*x)
        preprocessed_img = torch.stack(x, dim=0)
        x = self.model.image_encoder(preprocessed_img)

        if return_preprocessed:
            return x, preprocessed_img, paddings
        else:
            return x

    def sam_preprocess(self, x: torch.Tensor, return_paddings=False):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.model.pixel_mean) / self.model.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if return_paddings:
            return x, (padh, padw)
        else:
            return x

    def sam_preprocessing(self, x):
        x = [torchvision.transforms.ToPILImage()(img) for img in x]
        x = [np.array(img) for img in x]
        x = [self.sam_transform.apply_image(img) for img in x]
        x = [
            torch.from_numpy(img).permute(2, 0, 1).contiguous().to(self.device)
            for img in x
        ]
        x = [self.sam_preprocess(img, return_paddings=True) for img in x]
        x, paddings = zip(*x)
        preprocessed_img = torch.stack(x, dim=0)
        return preprocessed_img, paddings
    
    @torch.no_grad()
    def generate_bounding_boxes(self, images, features):
        """
        Generates bounding boxes for the given images.
        """
        if "anchor" in self.cfg["cellfinder"]:
            processed_imgs, _ = self.sam_preprocessing(images)
            results = self.cellfinder.forward_inference(processed_imgs)
        else:
            results = self.cellfinder.forward_inference(images, features=features, device=self.device)
        
        boxes_per_heatmap = [x["boxes"] for x in results]
        pred_labels = [x["labels"] for x in results]
        pred_scores = [x["scores"] for x in results]

        if self.bbox_threshold > 0.0:
            boxes_per_heatmap = [
                box[pred_scores[idx] > self.bbox_threshold]
                for idx, box in enumerate(boxes_per_heatmap)
            ]

        return boxes_per_heatmap

    @torch.no_grad()
    def generate_embeddings(self, images, existing_embeddings=None, transform=True):
        """
        Generates embeddings for the given images or uses existing embeddings if provided.
        """
        if existing_embeddings is None:
            transformed_images = self.predict_transforms(images) if transform else images
            embeddings, _, paddings = self(
                transformed_images, 
                return_heatmaps=False, 
                multimask_output=False, 
                return_preprocessed=True
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
        self,
        images,
        coords_per_heatmap=None,
        boxes_per_heatmap=None,
        return_lower_level_comps=False,
        global_threshold=0,
        transform=True,
        gt_map=None,
        prompts=None,
        x=None,
        **kwargs
    ):
        assert self.mask_threshold > 0  # otherwise all pred. will be true-> no blobs
        # x = batch
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        # if len(images.shape) == 3:
        #     images = images.unsqueeze(0)
        if "anchor" in self.cfg["cellfinder"] and self.cfg["prompter"] == "cellfinder":
            scaling_factor = 1.0
        else:
            scaling_factor = 1024 / max(images[0].shape)
        if x is None:
            print("making new embeddings!")
            x, paddings = self.generate_embeddings(images, transform=transform)
        else:
            # compute paddings -- we need them later on.
            # NOTE (Rohit): We need to do some work to stop recomputing things every time, but let's get something
            # working first
            print("using provided embedding!")
            paddings = []
            IMG_SIZE = self.model.image_encoder.img_size
            for img in images:
                h, w = img.shape[-2:]
                scale = int(1024 / max(h, w))
                h, w = h * scale, w * scale
                paddings.append((IMG_SIZE - h, IMG_SIZE - w))

        if coords_per_heatmap is None and boxes_per_heatmap is None:
            features = x

            # TODO: change the workflow here, rn embedded twice
            boxes_per_heatmap = self.generate_bounding_boxes(images, features)

        for idx, el in enumerate(range(len(x))):
            rng = 0
            if "point" in prompts:
                coords_for_heatmap = coords_per_heatmap[idx]
                rng = len(coords_for_heatmap)
            elif "box" in prompts:
                boxes_per_heatmap = boxes_per_heatmap[idx]
                rng = len(boxes_per_heatmap)
            else:
                rng = max([len(boxes_per_heatmap), len(coords_per_heatmap)])
            low_masks = []
            low_masks_thresholded = []
            scores = []
            for coord_idx, coord in enumerate(range(rng)):
                if "point" in prompts:
                    coord = coords_for_heatmap[coord_idx]
                    coord = [cen * scaling_factor for cen in coord]
                if "box" in prompts:
                    bbox = boxes_per_heatmap[coord_idx]
                    bbox = [cen * scaling_factor for cen in bbox]

                if "point" in prompts:
                    input_point = torch.as_tensor(coord).unsqueeze(0).unsqueeze(0)
                    input_label = torch.ones(
                        (input_point.shape[0], input_point.shape[1]), dtype=torch.float
                    )
                if "box" in prompts:
                    input_box = torch.as_tensor(bbox).unsqueeze(0).unsqueeze(0)
                    input_label = torch.ones(
                        (input_box.shape[0], input_box.shape[1]), dtype=torch.float
                    )
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    # points=None,
                    points=(input_point.to(self.device), input_label.to(self.device))
                    if "point" in prompts
                    else None,
                    # boxes=None,
                    boxes=input_box.to(self.device) if "box" in prompts else None,
                    # boxes=torch.tensor([[[0, 20, 200, 100]]]).to(self.device),
                    masks=None,
                )

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=x[idx].unsqueeze(0).to(self.device),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                low_res_masks = low_res_masks.detach().cpu()

                # threshold based on iou predictions
                truefalse = iou_predictions[0][0] < self.iou_threshold
                if truefalse:
                    print("iou too low")
                    continue
                low_res_masks = self.model.postprocess_masks(
                    low_res_masks,
                    input_size=torch.tensor(
                        [1024 - paddings[idx][0], 1024 - paddings[idx][1]]
                    ).to(self.device),
                    original_size=images[idx].shape[-2:],
                )
                low_res_masks_thresholded = (
                    nn.Sigmoid()(low_res_masks[0, 0]) > self.mask_threshold
                )

                # low_res_masks_thresholded = low_res_masks[0, 0]
                low_res_masks_thresholded = low_res_masks_thresholded.numpy().astype(
                    np.uint8
                )

                res = low_res_masks[0, 0].detach().cpu().numpy()

                assert res.shape == images[idx].shape[1:]
                low_masks.append(res)
                low_res_masks_thresholded = low_res_masks_thresholded[
                    : images[idx].shape[1], : images[idx].shape[2]
                ]
                low_masks_thresholded.append(low_res_masks_thresholded)
                scores.append(float(iou_predictions[0][0].detach().cpu().numpy()))

        if low_masks == []:
            if return_lower_level_comps:
                return (
                    np.zeros_like(images[0].detach().cpu().numpy()).astype(np.uint8),
                    np.zeros_like(images[0].detach().cpu().numpy()).astype(np.uint8),
                    np.zeros_like(images[0].detach().cpu().numpy()).astype(np.uint8),
                    np.zeros_like(images[0].detach().cpu().numpy()).astype(np.uint8),
                    np.zeros_like(images[0].detach().cpu().numpy()).astype(np.uint8),
                )
            else:
                return np.zeros_like(images[0].detach().cpu().numpy())

        low_masks = np.stack(low_masks)
        thresholded_masks = np.stack(low_masks_thresholded)
        scores = np.stack(scores)

        # multiply each mask by its index
        thresholded_masks_summed = (
            thresholded_masks
            * np.arange(1, thresholded_masks.shape[0] + 1)[:, None, None]
        )
        # sum all masks, #TODO: double check if max is the right move here
        thresholded_masks_summed = np.max(thresholded_masks_summed, axis=0)
        # check if results_dir exists

        if return_lower_level_comps:
            return thresholded_masks_summed, thresholded_masks, low_masks, scores, x, boxes_per_heatmap
        else:
            return thresholded_masks_summed, x

    def predict_transforms(self, imgs):
        imgs = [datapoints.Image(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)

        return imgs
