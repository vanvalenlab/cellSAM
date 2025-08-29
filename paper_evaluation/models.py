import torch
import numpy as np
from cellpose import models
from cellpose.models import Cellpose, CellposeModel, SizeModel, size_model_path
from cellpose.core import assign_device

# channel mappings
channel_to_use_for_training = {"Grayscale": 0, "Blue": 3, "Green": 2, "Red": 1}
second_training_channel = {**channel_to_use_for_training, "None": 0}


class Cellpose_size(Cellpose):
    """Cellpose + SizeModel."""

    def __init__(self, model_type=None, pretrained_model=None, gpu=True, device=None):
        sdevice, gpu = assign_device(True, gpu)
        self.device, self.gpu = device or sdevice, gpu
        self.diam_mean = 17. if model_type and 'nuclei' in model_type else 30.

        # base Cellpose model
        self.cp = CellposeModel(
            device=self.device, gpu=self.gpu,
            model_type=model_type, pretrained_model=pretrained_model,
            diam_mean=self.diam_mean,
        )
        self.diam_labels = self.cp.diam_labels

        # set pretrained_size as attribute so .eval() can find it
        self.pretrained_size = (
            size_model_path(model_type)
            if model_type else pretrained_model.replace('.pt', '_size.npy')
        )

        # size model
        self.sz = SizeModel(
            device=self.device,
            pretrained_size=self.pretrained_size,
            cp_model=self.cp
        )


class CellPoseModel:
    BUILTIN_MAP = {
        "nuclei": ([3, 0], 0.0),
        "cyto": ([3, 2], 0.0),
        "cyto2": ([3, 2], 0.0),
        "cyto3": ([3, 2], 0.0),
        "tissuenet": ([3, 2], 30.0),
        "livecell": ([3, 0], 30.0),
        "tissuenet_cp3": ([3, 2], 30.0),
        "livecell_cp3": ([3, 0], 30.0),
        "yeast_PhC_cp3": ([3, 2], 30.0),
        "yeast_BF_cp3": ([3, 2], 30.0),
        "bact_phase_cp3": ([3, 2], 30.0),
        "bact_fluor_cp3": ([3, 2], 30.0),
        "deepbacs_cp3": ([3, 2], 30.0),
        "cyto2_cp3": ([3, 2], 30.0),
    }

    def __init__(self, cfg, dataset=None):
        self.cfg = cfg
        self.dataset = dataset
        self.channels, self.diam_labels, self.model = self._init_model()
        self.criteria = torch.nn.CrossEntropyLoss()

    def _init_model(self):
        # pretrained path
        if self.cfg["model"]["pretrain"]:
            chans = [self.cfg["cellpose"]["chan"], self.cfg["cellpose"]["chan2"]]
            pm = self.cfg["cellpose"]["pretrained_model"]

            if self.cfg["cellpose"]["with_size"]:
                return chans, 0.0, Cellpose_size(
                    pretrained_model=pm, gpu=not self.cfg["is_debug"]
                )
            model = models.CellposeModel(pretrained_model=pm, gpu=not self.cfg["is_debug"])
            return chans, model.diam_labels, model

        # built-in
        if mtype not in self.BUILTIN_MAP:
            raise ValueError(f"Unsupported model_type {mtype}")

        chans, diam = self.BUILTIN_MAP[mtype]
        if mtype == "nuclei" and self.dataset in ["tissuenet_wholecell", "tissuenet_nuclear"]: # for tissuenet datasets nuclei detection, only use channel 2
            chans = [2, 0]
        ModelCls = models.CellposeModel if diam > 0 else models.Cellpose
        return chans, diam, ModelCls(model_type=mtype, gpu=not self.cfg["is_debug"])

    def predict(self, batch):
        masks, *_ = self.model.eval(batch, batch_size=8, channels=self.channels, diameter=None)
        return [np.expand_dims(pred, -1) for pred in masks]
