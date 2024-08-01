""" Napari plugin for cellSAM segmentation. 

Based on the Segment Anything napari implementation at royerlab/napari-segment-anything
"""

from typing import Any, Generator, Optional

import napari
import numpy as np
from magicgui.widgets import ComboBox, Container, PushButton, create_widget
from napari.layers import Image, Shapes
from napari.layers.shapes._shapes_constants import Mode
from napari.utils import DirectLabelColormap

from skimage import color, util
import torch
from qtpy.QtWidgets import QApplication
from warnings import warn


from cellSAM import get_model
from cellSAM.utils import normalize_image, fill_holes_and_remove_small_masks


def get_rotating_color_map():
    # Define 10 distinct colors
    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ]
    return napari.utils.CyclicLabelColormap(colors=colors, background_value=0)

class CellSAMWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._viewer = viewer
        self._cellsam_model = get_model()

        self._channels = []

        # Hacky to deal with a known magicgui issue where channels don't update.
        # For some reason, I can't just directly pass a list to choices.
        def _get_channels(*args):
            return self._channels

        # Initialize channel selection ComboBox, initially hidden
        self.nuclear_channel_selector = ComboBox(
            label="Select nuclear channel:", choices=_get_channels
        )
        self.nuclear_channel_selector.visible = (
            False  # Hide until an appropriate image is loaded
        )
        self.nuclear_channel_selector.enabled = (
            True  # Hide until an appropriate image is loaded
        )

        self.wc_channel_selector = ComboBox(
            label="Select wholecell channel:", choices=_get_channels
        )
        self.wc_channel_selector.visible = (
            False  # Hide until an appropriate image is loaded
        )
        self.wc_channel_selector.enabled = (
            True  # Hide until an appropriate image is loaded
        )

        self.append(self.nuclear_channel_selector)
        self.append(self.wc_channel_selector)

        self._viewer = viewer
        self._viewer.layers.events.inserted.connect(self._on_layer_added)

        self._im_layer_widget = create_widget(annotation=Image, label="Image:")
        self._im_layer_widget.changed.connect(self._load_image)
        self.append(self._im_layer_widget)

        # segment everything
        self._segment_all_btn = PushButton(
            text="Segment All",
            tooltip="Segment all bounding boxes.",
        )
        self._segment_all_btn.clicked.connect(self._on_segment_all)
        self.append(self._segment_all_btn)

        # Add a new layer for the segmentation overlay
        self._segmentation_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int),
            name="Segmentation Overlay",
        )

        self._confirm_mask_btn = PushButton(
            text="Confirm Annot.",
            enabled=False,
            tooltip="Press C to confirm annotation.",
        )
        self._confirm_mask_btn.changed.connect(self._on_confirm_mask)
        self.append(self._confirm_mask_btn)

        self._cancel_annot_btn = PushButton(
            text="Cancel Annot.",
            enabled=False,
            tooltip="Press X to cancel annotation.",
        )
        self._cancel_annot_btn.changed.connect(self._cancel_annot)
        self.append(self._cancel_annot_btn)

        self._clear_seg_btn = PushButton(
            text="Clear mask",
            enabled=False,
        )
        self._clear_seg_btn.changed.connect(self._clear_segmentation)
        self.append(self._clear_seg_btn)

        self._reset_btn = PushButton(
            text="Reset",
            enabled=True,
            tooltip="Run this before loading a new image!",
        )
        self._reset_btn.changed.connect(self._reset)
        self.append(self._reset_btn)

        cmap = DirectLabelColormap(color_dict=dict(zip(range(1, 3501), [[255,0,0]] * 3500)))

        self._mask_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int),
            name="Drawn masks",
            colormap=cmap
        )

        self._mask_layer.contour = 2

        self._boxes_layer = self._viewer.add_shapes(
            name="Bounding boxes",
            face_color="transparent",
            edge_color="green",
            edge_width=2,
        )
        self._boxes_layer.mouse_drag_callbacks.append(self._on_shape_drag)

        self._image: Optional[np.ndarray] = None

        self._viewer.bind_key("C", self._on_confirm_mask)
        self._viewer.bind_key("X", self._cancel_annot)

        self._norm_image = None
        self._embedding = None

    def _on_layer_added(self, event):
        """Callback for when a layer is added to the viewer"""
        layer = event.value
        if isinstance(layer, Image):
            self._process_new_image(layer)

    def update_channel_selector(self, image):
        if image.ndim == 2:
            # Grayscale image, assume one channel
            # need to be able to disgintuish between the two cases
            self._channels = ["None"]
            for selector in [self.nuclear_channel_selector, self.wc_channel_selector]:
                selector.enabled = False  # Disable if not needed for single channel
                selector.visible = False
        elif image.ndim == 3:
            # Color or multi-channel image
            num_channels = image.shape[2]
            self._channels = [f"Channel {i}" for i in range(num_channels)] + ["None"]
            for selector in [self.nuclear_channel_selector, self.wc_channel_selector]:
                selector.enabled = True
                selector.visible = True
        else:
            raise ValueError("Unsupported image dimensionality")

    def _on_segment_all(self):

        if self._norm_image is None:
            return
        # Segment using all bounding boxes

        inp = self._norm_image
        # mean pool if no channels are selected. This should do nothing to grayscale images
        # if one of the channels is provided, then use that channel

        nuc_ch, wc_ch = (
            self.nuclear_channel_selector.value,
            self.wc_channel_selector.value,
        )
        if nuc_ch == "None" and wc_ch == "None":
            nuc_cid, wc_cid = None, None
        else:
            nuc_cid = self._channels.index(nuc_ch)
            wc_cid = self._channels.index(wc_ch)

        inp = np.zeros((3, *self._norm_image.shape[:2]), dtype=np.float32)

        if nuc_cid is None and wc_cid is None:
            inp[2] = self._norm_image.mean(axis=2)
        elif nuc_cid is not None and wc_cid is not None:
            inp[1, ...] = self._norm_image[:, :, nuc_cid]
            inp[2, ...] = self._norm_image[:, :, wc_cid]
        elif nuc_cid is not None and wc_cid is None:
            inp[2, ...] = self._norm_image[:, :, nuc_cid]
        elif nuc_cid is None and wc_cid is not None:  # this case might be problematic
            inp[2, ...] = self._norm_image[:, :, wc_cid]

        inp = torch.from_numpy(inp).unsqueeze(0)

        preds, _, x, _ = self._cellsam_model.predict(
            inp.to(self._device),
            x=self._embedding,
            boxes_per_heatmap=None,
            device="cpu",
        )

        if preds is None:
            warn("No cells detected!")
            return

        mask = fill_holes_and_remove_small_masks(preds, min_size=25)

        # Update the segmentation layer
        self._segmentation_layer.data = mask
        self._segmentation_layer.colormap = get_rotating_color_map()  # Apply the color map
        self._segmentation_layer.visible = True
        self._clear_seg_btn.enabled = True

        self._embedding = x

    def _process_new_image(self, im_layer: Image):
        image = im_layer.data
        self._image = image
        norm_image = normalize_image(image)
        self._norm_image = norm_image

        # Initialize or update mask layer to match new image dimensions
        if hasattr(self, '_mask_layer'):
            self._mask_layer.data = np.zeros(image.shape[:2], dtype=int)
        else:
            self._mask_layer = self._viewer.add_labels(
                np.zeros(image.shape[:2], dtype=int),
                name="SAM mask",
                color=dict(zip(range(1, 3501), ["red"] * 3500)),
                contour=2
            )

        # Initialize or update segmentation overlay layer to match new image dimensions
        if hasattr(self, '_segmentation_layer'):
            self._segmentation_layer.data = np.zeros(image.shape[:2], dtype=int)
        else:
            self._segmentation_layer = self._viewer.add_labels(
                np.zeros(image.shape[:2], dtype=int),
                name="Segmentation Overlay"
            )

        self.update_channel_selector(image)
        if image.ndim == 2:
            # grayscale
            image = image[:, :, None]
        elif image.ndim == 3:
            c_idx = np.argmin(image.shape)
            axes = [ax for ax in range(image.ndim) if ax != c_idx] + [c_idx]
            image = image.transpose(axes)
        else:
            raise ValueError("Image must be 2D, or 2D with multiple channels.")

        norm_image = normalize_image(image)

        self._norm_image = norm_image
        self._image = util.img_as_ubyte(image)
        self._mask_layer.data = np.zeros(self._image.shape[:2], dtype=int)

        # After processing the image, move it to the end of the layer list
        current_index = self._viewer.layers.index(im_layer)
        target_index = 0
        self._viewer.layers.move(current_index, target_index)
        QApplication.processEvents()  # Process all pending GUI events, might help update the state

    def _load_image(self, im_layer: Optional[Image]) -> None:
        if im_layer is None:
            return
        image = im_layer.data
        if not im_layer.rgb:
            image = color.gray2rgb(image)
        elif image.shape[-1] == 4:
            image = color.rgba2rgb(image)

        if np.issubdtype(image.dtype, np.floating):
            image = image - image.min()
            image = image / image.max()

        self._image = util.img_as_ubyte(image)
        self._mask_layer.data = np.zeros(self._image.shape[:2], dtype=int)
        self.update_channel_selector(self._image)

    def _on_interactive_run(self, _: Optional[Any] = None) -> None:
        boxes = self._boxes_layer.data

        if self._im_layer_widget.value is None or not boxes:
            return

        formatted_boxes = np.array(
            [np.stack([box.min(axis=0), box.max(axis=0)], axis=0) for box in boxes]
        )
        formatted_boxes = np.flip(formatted_boxes, axis=-1).reshape(-1, 4)

        inp = self._norm_image
        # mean pool if no channels are selected. This should do nothing to grayscale images
        # if one of the channels is provided, then use that channel

        nuc_ch, wc_ch = (
            self.nuclear_channel_selector.value,
            self.wc_channel_selector.value,
        )
        if nuc_ch == "None" and wc_ch == "None":
            nuc_cid, wc_cid = None, None
        else:
            nuc_cid = self._channels.index(nuc_ch)
            wc_cid = self._channels.index(wc_ch)
        inp = np.zeros((3, *self._norm_image.shape[:2]), dtype=np.float32)

        if nuc_cid is None and wc_cid is None:
            inp[2] = self._norm_image.mean(axis=2)
        elif nuc_cid is not None and wc_cid is not None:
            inp[1, ...] = self._norm_image[:, :, nuc_cid]
            inp[2, ...] = self._norm_image[:, :, wc_cid]
        elif nuc_cid is not None and wc_cid is None:
            inp[2, ...] = self._norm_image[:, :, nuc_cid]
        elif nuc_cid is None and wc_cid is not None:  # this case might be problematic
            inp[2, ...] = self._norm_image[:, :, wc_cid]

        inp = torch.from_numpy(inp).unsqueeze(0)

        preds, _, x, _ = self._cellsam_model.predict(
            inp.to(self._device),
            x=self._embedding,
            boxes_per_heatmap=torch.tensor(formatted_boxes)
            .unsqueeze(0)
            .to(self._device),
            device="cpu",
        )
        if preds is None:
            warn("No cells detected!")
        else:
            preds = fill_holes_and_remove_small_masks(preds, min_size=25)
            self._mask_layer.data = preds
            self._confirm_mask_btn.enabled = True
            self._cancel_annot_btn.enabled = True
            self._clear_seg_btn.enabled = True

            self._embedding = x

    def _on_shape_drag(self, _: Shapes, event) -> Generator:
        if self._boxes_layer.mode != Mode.ADD_RECTANGLE:
            return
        yield
        while event.type == "mouse_move":
            yield
        self._on_interactive_run()

    def _on_confirm_mask(self, _: Optional[Any] = None) -> None:
        if self._image is None:
            return

        labels = self._segmentation_layer.data
        mask = self._mask_layer.data

        labels[np.nonzero(mask)] = labels.max() + 1
        self._segmentation_layer.data = labels
        self._segmentation_layer.colormap = get_rotating_color_map()  # Apply the color map
        self._segmentation_layer.visible = True

        self._clear_seg_btn.enabled = True

        self._cancel_annot()

    def _cancel_annot(self, _: Optional[Any] = None) -> None:
        # NOTE: this should be self._boxes_layer.data = [], but napari is somewhat fragile and
        # handles this poorly on the backend. This is a hack but seems to work fine. 
        if len(self._boxes_layer.data):
            self._boxes_layer.data = [np.zeros_like(box) for box in self._boxes_layer.data]
        if self._mask_layer.data.any():
            self._mask_layer.data = np.zeros_like(self._mask_layer.data)

    def _clear_segmentation(self, _: Optional[Any] = None) -> None:
        self._segmentation_layer.data = np.zeros_like(self._segmentation_layer.data)
        self._confirm_mask_btn.enabled = False
        self._cancel_annot_btn.enabled = False
        self._clear_seg_btn.enabled = False
    

    def _reset(self, _: Optional[Any] = None) -> None:
        self._segmentation_layer.data = np.zeros_like(self._segmentation_layer.data)

        self._boxes_layer.data = [np.zeros_like(box) for box in self._boxes_layer.data]
        self._mask_layer.data = np.zeros_like(self._mask_layer.data)
        self._embedding = None
        self._image = None
        self._norm_image = None

        for layer in list(self._viewer.layers):  # Use list to avoid modifying the iterable during iteration
            if isinstance(layer, napari.layers.Image):
                self._viewer.layers.remove(layer)


        self._confirm_mask_btn.enabled = False
        self._cancel_annot_btn.enabled = False
        self._clear_seg_btn.enabled = False
        self._reset_btn.enabled = True
    
