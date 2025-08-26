import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class MixDataset(Dataset):
    def __init__(self, split, dataset=None, crop_size=512, root_dir=None,
                 data_type="npy", normalize=False, CLAHE=False, duplicate=False, replacement=False):
        assert split in ["train", "test", "val", "everything"], f"split={split} must be one of train, val, test or everything."
        assert data_type in ["npy", "tiff"], f"data_type={data_type} must be one of npy or tiff."

        self.split = split
        self.crop_size = crop_size
        self.normalize = normalize
        self.CLAHE = CLAHE
        self.dataset_size = []

        if crop_size > 0:
            split_ext = f"{split}_crop_{crop_size}"
        elif crop_size == 0:
            split_ext = split
        else:
            raise ValueError(f'crop_size = {crop_size} must be >= 0!')

        root_dir = Path(root_dir) if root_dir else Path("./data/dataset/")
        img_ext = "X.npy" if data_type == "npy" else "_img.tiff"
        mask_ext = "y.npy" if data_type == "npy" else "_masks.tiff"

        dataset = dataset or "*"
        if isinstance(dataset, str):
            all_imgs = list((root_dir / split_ext).glob(f"{dataset}/*{img_ext}"))
        elif isinstance(dataset, list):
            all_imgs = []
            for ds in dataset:
                imgs = list((root_dir / split_ext).glob(f"{ds}/*{img_ext}"))
                all_imgs.extend(imgs)
                self.dataset_size.append(len(imgs))
        else:
            raise ValueError("dataset must be str or list of str.")

        if not all_imgs:
            raise ValueError(f"No images found in {root_dir / split_ext / dataset}!")

        self.all_imgs = all_imgs
        self.all_masks = [img.with_suffix("").with_suffix(".y.npy") for img in all_imgs]
        if split == "train" and self.dataset_size:
            weights = [1.0 / x for x in self.dataset_size]
            sample_weights = []
            for w, length in zip(weights, self.dataset_size):
                sample_weights.extend([w] * length)
            self.sample_weights = torch.tensor(sample_weights, dtype=torch.double)
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
                self.sample_weights, len(self.all_imgs), replacement=replacement)
        elif split == "val":
            idcs = list(range(len(self.all_imgs)))
            random.shuffle(idcs)
            self.all_imgs = [self.all_imgs[i] for i in idcs]
            self.all_masks = [self.all_masks[i] for i in idcs]

    def __getitem__(self, idx):
        X, y = np.load(self.all_imgs[idx]), np.load(self.all_masks[idx])
        
        if X.shape[0] != 3:
            X = X.transpose(2, 0, 1)
        if X.shape[0] == 2:
            X = np.concatenate([np.zeros_like(X[0])[np.newaxis, :, :], X], axis=0)
        assert X.ndim == 3 and X.shape[0] == 3, f"X shape invalid: {X.shape}"
        assert y.ndim == 3, f"y shape invalid: {y.shape}"
        assert X.shape[1:] == y.shape[1:], f"Shape mismatch: X{X.shape[1:]} != y{y.shape[1:]}"
        
        return torch.from_numpy(X), torch.from_numpy(y), self.all_imgs[idx]

    def get_metadata(self, idx):
        return self.all_imgs[idx]

    def __len__(self):
        return len(self.all_imgs)
