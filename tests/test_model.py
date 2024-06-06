import os
import pytest
from pathlib import Path
import imageio.v3 as iio
from warnings import warn
from scipy.stats import hmean
import numpy as np
import torch
from cellSAM import segment_cellular_image, get_model
from cellSAM.utils import normalize_image

def _f1_score(pred, target):
    """ Adapted from DeepCell. This does not do matching before calculating the F1 score,
    but is fine for a quick test. 
    """

    pred_sum = np.count_nonzero(pred)
    target_sum = np.count_nonzero(target)

    # Calculations for IOU
    intersection = np.count_nonzero(np.logical_and(pred, target))
    union = np.count_nonzero(np.logical_or(pred, target))
    recall = intersection / target_sum 
    precision = intersection /pred_sum 

    return hmean([recall, precision])

def test_segment_cellular_image_slow():
    current_pth = Path(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warn('Using CPU for testing. This may take a while.')
    for fname in ['YeaZ.png', 'YeastNet.png', 'ep_micro.png', 'tissuenet.png']:
        img = iio.imread(current_pth.parent.parent / 'sample_imgs' / fname)
        mask = segment_cellular_image(img, model=None, normalize=True, device=device, fast=False)[0]
        pred = np.load(current_pth.parent.parent / 'sample_imgs' / (fname.replace('.png', '_pred.npy')))
        assert _f1_score(pred, mask) > 0.95 

def test_segment_cellular_image_fast():
    current_pth = Path(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warn('Using CPU for testing. This may take a while.')
    for fname in ['YeaZ.png', 'YeastNet.png', 'ep_micro.png', 'tissuenet.png']:
        img = iio.imread(current_pth.parent.parent / 'sample_imgs' / fname)
        mask = segment_cellular_image(img, model=None, normalize=True, device=device, fast=True)[0]
        pred = np.load(current_pth.parent.parent / 'sample_imgs' / (fname.replace('.png', '_pred.npy')))
        assert _f1_score(pred, mask) > 0.95 

# a bit easier...
if __name__ == '__main__':
    test_segment_cellular_image_slow()
    test_segment_cellular_image_fast()