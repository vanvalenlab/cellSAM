"""
Evaluate model performance on the NeurIPS validation dataset.

Note:
- Intended for development purposes only.
- Pre- and post-processing steps are omitted; results are not directly comparable to those reported in the original paper.

Expected Output:
- Average F1 score ~0.88
- Runtime: ~44 second on a single GPU (4090)
"""
import os
import numpy as np
import torch
import time
from tqdm import tqdm
from cellSAM.utils import f1_score
from cellSAM import segment_cellular_image, get_model


def load_data(path: str, fast=False):
    all_files = os.listdir(path)
    images = sorted([f for f in all_files if f.endswith('.X.npy')])
    masks = sorted([f for f in all_files if f.endswith('.y.npy')])

    # skip last file (huge)
    images = images[:-1]
    masks = masks[:-1]

    if fast:
        idx = np.random.choice(len(images), 10)
        images = [images[i] for i in idx]
        masks = [masks[i] for i in idx]

    images = [np.load(os.path.join(path, img)) for img in images]
    masks = [np.load(os.path.join(path, msk)) for msk in masks]

    return images, masks


def evaluate_model(images, masks, model, device='cuda'):
    f1_scores = []
    for img, true_mask in tqdm(zip(images, masks)):
        pred_mask = segment_cellular_image(img, model=model, device=device)[0]
        score = f1_score(true_mask, pred_mask)
        f1_scores.append(score)
    return np.mean(f1_scores)


def main():
    # Set to True to speed up evaluation (only picks 10 rnd samples)
    FAST = True

    torch.manual_seed(0)
    np.random.seed(0)

    starttime = time.time()
    data_dir = '/data/user-data/rdilip/cellSAM/dataset/val/neurips_fixed'
    model_path = '/data/AllCellData/results/new_finetuned/mixed_everything_goodlive_neurips_tuning_aug_ft_box/SAM_groundtruth_boxPrompt_everything_with_good_livecell_neurips_tuning_aug_train.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_path).to(device)

    images, masks = load_data(data_dir, fast=FAST)
    avg_f1_score = evaluate_model(images, masks, model, device=device)

    print(f"Average F1 score: {avg_f1_score:.4f}")
    print(f"Runtime: {time.time() - starttime:.2f} seconds")


if __name__ == "__main__":
    main()
