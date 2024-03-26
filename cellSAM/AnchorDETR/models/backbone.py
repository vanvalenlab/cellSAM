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
Backbone modules.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from segment_anything import sam_model_registry
from ...AnchorDETR.util.misc import NestedTensor, is_main_process


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = []
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer,
        )
        assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class SAMBackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        return_interm_layers: bool,
        only_neck: bool,
        freeze_backbone: bool = False,
        sam_vit: str = "vit_h",
    ):
        super().__init__()
        # TODO: adjust later for SAM
        # for name, parameter in backbone.named_parameters(): #TODO: adjust later for SAM
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if freeze_backbone:
            for name, parameter in backbone.named_parameters():
                parameter.requires_grad_(False)
        if only_neck:
            for name, parameter in backbone.named_parameters():
                if "neck" in name:
                    parameter.requires_grad_(True)
                else:
                    parameter.requires_grad_(False)
        if return_interm_layers:
            raise NotImplementedError
        else:
            return_layers = {"blocks": "0"}
            self.strides = [4]  # TODO: just changed, see if it makes sense
            if sam_vit == "vit_h":
                self.num_channels = [1280]
            elif sam_vit == "vit_b":
                self.num_channels = [768]
            elif sam_vit == "vit_l":
                self.num_channels = [1024]
            else:
                raise NotImplementedError
        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        x = self.body(tensor_list.tensors)
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out = NestedTensor(x, mask)
        return [out]


class ModifiedImageEncoderViT(nn.Module):
    def __init__(self, original_model):
        super(ModifiedImageEncoderViT, self).__init__()

        # Extract all layers up to but not including 'neck'
        self.patch_embed = original_model.patch_embed
        self.blocks = original_model.blocks
        self.pos_embed = original_model.pos_embed

    def forward(self, x):
        # Apply patch embedding
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Pass through blocks
        for block in self.blocks:
            x = block(x)

        return x.permute(0, 3, 1, 2)


class SAMBackbone(SAMBackboneBase):
    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool = False,
        dilation: bool = False,
        only_neck: bool = False,
        freeze_backbone: bool = False,
        sam_vit: str = "vit_h",
    ):
        backbone = sam_model_registry[sam_vit]()
        backbone = backbone.image_encoder
        backbone = ModifiedImageEncoderViT(backbone)
        super().__init__(
            backbone,
            train_backbone,
            return_interm_layers,
            only_neck,
            freeze_backbone,
            sam_vit,
        )
        if dilation:  # TODO: check if applicable here
            self.strides[-1] = self.strides[-1] // 2


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if args.backbone == "SAM":
        backbone = SAMBackbone(
            args.backbone,
            train_backbone,
            return_interm_layers,
            args.dilation,
            args.only_neck,
            args.freeze_backbone,
            args.sam_vit,
        )
    else:
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation
        )
    return backbone
