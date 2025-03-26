""" Main class for CellSAM data

Prereqs: run prepare.py, it's not structured as a command line tool yet, maybe if this becomes
an actual pipeline and not me hacking away.

@author: rohit dilip
@date: 11/28/2023
"""
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from skimage.io import imread
from torch.utils.data import Dataset


def histogram_normalization(image, kernel_size=128):
    """
    Modified from deepcell_toolbox.processing: histogram_normalization
    Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    Args:
        image (numpy.array): numpy array of phase image data. shape: H, W, C
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    assert isinstance(image, np.ndarray), "image should be a numpy array"
    assert image.ndim == 3, "image should be 3D"

    image = image.astype("float32")

    for channel in range(image.shape[-1]):
        X = image[..., channel]
        sample_value = X[(0,) * X.ndim]
        if (X == sample_value).all():
            image[..., channel] = np.zeros_like(X)
            continue

        X = rescale_intensity(X, out_range=(0.0, 1.0))
        X = equalize_adapthist(X, kernel_size=kernel_size)
        image[..., channel] = X

    return image


class MixDataset(Dataset):
    def __init__(
            self,
            split,
            dataset=None,
            crop_size=512,
            root_dir=None,
            data_type: str = "npy",
            normalize=False,
            CLAHE=False,
            duplicate=False,
            replacement=False,
    ):
        assert split in [
            "train",
            "test",
            "val",
            "everything",
        ], f"split={split} must be one of train, val, test or everything."
        assert data_type in [
            "npy",
            "tiff",
        ], f"data_type={data_type} must be one of npy or tiff."
        # TODO: tiff not implemented yet

        self.split = split
        self.duplicate = duplicate
        self.crop_size = crop_size
        self.datatype = data_type
        self.flattened = False
        self.normalize = normalize  # if normalize the image to [0, 1]
        self.CLAHE = CLAHE  # if use CLAHE to enhance the contrast of the image
        self.dataset_size = []

        if crop_size > 0:
            split_ext = f"{split}_crop_{crop_size}"
        elif crop_size == 0:  # use original image
            split_ext = split
        else:
            raise ValueError(f'crop_size = {crop_size} must be a int > 0!')

        if root_dir is None:
            root_dir = Path("./data/dataset/")
        else:
            root_dir = Path(root_dir)
        if data_type == "npy":
            img_ext = "X.npy"
            mask_ext = "y.npy"
        elif data_type == "tiff":
            img_ext = "_img.tiff"
            mask_ext = "_masks.tiff"

        # add dataset
        if dataset is None or dataset == "":
            dataset = "*"
        if isinstance(dataset, str):
            all_imgs = list((root_dir / split_ext).glob(f"{dataset}/*{img_ext}"))
        elif isinstance(dataset, list):
            if split in ['train', 'val', 'test']:
                all_imgs = []
                for ds in dataset:
                    imgs = list(
                        (root_dir / split_ext).glob(f"{ds}/*{img_ext}")
                    )
                    all_imgs = all_imgs + imgs
                    # self.dataset_size[ds] = len(imgs)
                    self.dataset_size.append(len(imgs))
            elif split == 'everything':
                all_imgs = []
                for sub_split in ['train', 'val',
                                  'test']:  # TODO: this isn't working! So fine-tuning on this also doesn't work as expected.
                    for ds in dataset:
                        all_imgs = all_imgs + list(
                            (root_dir / split_ext).glob(f"{ds}/*{img_ext}")
                        )
            else:
                raise ValueError("not proper split name")
        else:
            raise ValueError("Invalid image format. Must be str or list of str.")

        print(f"MixDataset: {len(all_imgs)} images found in dataset {dataset}.")
        if len(all_imgs) == 0:
            raise ValueError(
                f"No images found in {root_dir / split_ext / dataset}! You may have used an incorrect root path or dataset name."
            )

        if self.duplicate:
            all_imgs = duplicate_sample_lst_with_dataset_balancing(all_imgs)  # TODO: check, deprecated?
        if data_type == "npy":
            all_masks = [img.with_suffix("").with_suffix(".y.npy") for img in all_imgs]
        elif data_type == "tiff":
            all_masks = [Path(str(img).replace(img_ext, mask_ext)) for img in all_imgs]

        self.all_imgs = all_imgs
        self.all_masks = all_masks
        if split == "train":
            total_images = len(self.all_imgs)
            # inverse
            weights = [1.0/x for x in self.dataset_size]
            sample_weights = []
            for w, length in zip(weights, self.dataset_size):
                sample_weights += [w] * length

            self.sample_weights = sample_weights
            self.sample_weights = torch.from_numpy(np.array(self.sample_weights)).double()
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
                self.sample_weights, total_images, replacement=replacement
            )
        if split == "val":
            # shuffle the dataset using random permutation
            idcs = list(range(len(self.all_imgs)))
            random.shuffle(idcs)
            self.all_imgs = [self.all_imgs[i] for i in idcs]
            self.all_masks = [self.all_masks[i] for i in idcs]

    def __getitem__(self, idx):
        if '4599066422394007024' in str(self.all_imgs[idx]):
            idx = idx+1
        if self.datatype == "npy":
            X, y = np.load(self.all_imgs[idx]), np.load(self.all_masks[idx])
        elif self.datatype == "tiff":
            X, y = imread(self.all_imgs[idx]), imread(self.all_masks[idx])

            if X.shape[-1] == 3:  # change HWC to CHW
                X = X.transpose(2, 0, 1)
            if y.ndim == 2:  # change HW to CHW
                y = y[np.newaxis, :, :]
            if y.shape[-1] == 1:  # change HWC to CHW
                y = y.transpose(2, 0, 1)

        if 'Gendarme' in str(self.all_imgs[idx]):
            X = X[[0, 2, 1], :, :]

        # check that the shapes are correct
        assert X.ndim == 3, f"X.ndim = {X.ndim} != 3"
        assert X.shape[0] == 3, f"X.shape[0] = {X.shape[0]} != 3"
        assert y.ndim == 3, f"y.ndim = {y.ndim} != 3"
        assert (
                X.shape[1:] == y.shape[1:]
        ), f"X.shape[1:] = {X.shape[1:]} != y.shape[1:] = {y.shape[1:]}"

        return torch.from_numpy(X), torch.from_numpy(y), self.all_imgs[idx]

    def get_metadata(self, idx):
        return self.all_imgs[idx]

    def __len__(self):
        return len(self.all_imgs)

    def flatten_training_splits(self, n_img=None, seed=42):
        """
        This will overwrite the current dataset to contain n_img images from
        each dataset. If n_img is None, some extra computation is done to select the minimum number
        from each dataset.

        The current behavior will take n_img images from each dataset, unless the dataset contains
        fewer than that number (e.g., some of the deepbacs subdatasets). You should initialize the
        dataset then call this. It will throw a warning if you call it multiple times.

        >> ds = CellsamDataset('train')
        >> print(len(ds))
        >> ds.flatten_training_splits(n_img=10)
        >> print(len(ds))
        >> ds.flatten_training_splits(n_img=20) # this will throw a warning.

        """
        if self.flattened:
            warnings.warn(
                "The current version flatten_training_splits is stateful. You've run it multiple times, which"
                " is probably not the desired behaior. Reinitialize the dataset and run it once. The stateful"
                " behavior will be deprecated (probably)."
            )
        if self.duplicate:
            raise ValueError("duplicate cannot be set to True if you call self.flatten_training_splits")
        self.flattened = True
        random.seed(seed)
        # meh i don't like this but itll work, we're on a schedule
        dataset_names = set([img.parent.name for img in self.all_imgs])
        files_by_dataset = {}
        for dataset_name in dataset_names:
            files_by_dataset[dataset_name] = [
                img for img in self.all_imgs if img.parent.name == dataset_name
            ]
        n_img_by_dataset = {}
        for dataset_name in dataset_names:
            n_img_by_dataset[dataset_name] = min(
                n_img, len(files_by_dataset[dataset_name])
            )
        print(n_img_by_dataset)

        self.all_imgs = [
            _fname
            for dataset_name in dataset_names
            for _fname in random.sample(
                files_by_dataset[dataset_name], k=n_img_by_dataset[dataset_name]
            )
        ]
        self.all_masks = [
            img.with_suffix("").with_suffix(".y.npy") for img in self.all_imgs
        ]


def duplicate_sample_lst_with_dataset_balancing(all_samples):
    """ Duplicate sample list with consistency with Qilin's version. We should perform sampling
    in a different way (for one, not hard coded), but there's value in pushing this out as 
    quickly as possible. This has datatype and dataset balancing, but not subdataset balancing.

    """
    import warnings
    msg = """
    You are using a hardcoded version of sample duplication! This is fragile and will break
    if you change any of the datasets. It is set as is for consistency with existing dataloaders.
    """
    warnings.warn(msg)

    dataset_to_num_of_copy_dict = {
        '2b_brightfield_dataset': 7,
        '2b_fluorescence_dataset': 7,
        '2c_e_coli': 7,
        '2d_1_SplineDist_dataset': 7,
        '2d_2_b_subtilis': 7,
        '2e_e_coli': 7,
        's2_stardist': 7,
        'bact_fluor': 13,
        'bact_phase': 13,
        '3T3_ep_microscopy': 1,
        'A549_ep_microscopy': 1,
        'CHO_ep_microscopy': 1,
        'HEK293_ep_microscopy': 1,
        'HeLa-S3_ep_microscopy': 1,
        'HeLa_ep_microscopy': 1,
        'PC3_ep_microscopy': 1,
        'RAW264_ep_microscopy': 1,
        'cpm15': 16,
        'cpm17': 16,
        'kumar': 16,
        'monusac': 16,
        'monuseg': 16,
        'nuinsseg': 16,
        'tnbc': 16,
        'Gendarme_BriFi': 36,
        'YeaZ': 11,
        'YeastNet': 20,
        'cellpose': 13,
        'nuc_seg_dsb': 23,
        'tissuenet_wholecell': 8,
    }
    all_samples_duplicated = []
    for sample in all_samples:
        ds = sample.parent.name
        all_samples_duplicated = all_samples_duplicated + dataset_to_num_of_copy_dict[ds] * [sample]
    return all_samples_duplicated


def duplicate_sample_lst(all_samples):
    """Constructs a weighted sampler based on a list of paths to samples. 
    Implicitly assumes that the parent directory of the samples is the name 
    of the dataset. Will throw an error if the dataset isn't reflected in the
    internal mapper to datatype. Sampling is equally weighted across data 
    modalities (Tissue, Bacteria, Cell culture, H&E, Yeast, Nuclear, Mixed), 
    then weighted by dataset.
    """
    dataset_to_datatype = {
        "tissuenet_wholecell": "Tissue",
        "2b_brightfield_dataset": "Bacteria",
        "2b_fluorescence_dataset": "Bacteria",
        "2c_e_coli": "Bacteria",
        "2d_1_SplineDist_dataset": "Bacteria",
        "2d_2_b_subtilis": "Bacteria",
        "2e_e_coli": "Bacteria",
        "s2_stardist": "Bacteria",
        "bact_fluor": "Bacteria",
        "bact_phase": "Bacteria",
        "3T3_ep_microscopy": "Cell culture",
        "A549_ep_microscopy": "Cell culture",
        "CHO_ep_microscopy": "Cell culture",
        "HEK293_ep_microscopy": "Cell culture",
        "HeLa_ep_microscopy": "Cell culture",
        "HeLa-S3_ep_microscopy": "Cell culture",
        "PC3_ep_microscopy": "Cell culture",
        "RAW264_ep_microscopy": "Cell culture",
        "Gendarme_BriFi": "Cell culture",
        "cellpose": "Cell culture",
        "cpm15": "H&E",
        "cpm17": "H&E",
        "kumar": "H&E",
        "monusac": "H&E",
        "monuseg": "H&E",
        "nuinsseg": "H&E",
        "tnbc": "H&E",
        "YeastNet": "Yeast",
        "YeaZ": "Yeast",
        "nuc_seg_dsb": "Mixed",
    }
    all_samples = [Path(pth) for pth in all_samples]
    root_dir = list(set([pth.parent.parent.parent for pth in all_samples]))

    if len(root_dir) != 1:
        raise ValueError("Multiple root directories found!")
    root_dir = root_dir[0]

    numel_by_datatype = defaultdict(int)
    for sample in all_samples:
        ds = sample.parent.name
        numel_by_datatype[dataset_to_datatype[ds]] += 1  # sample by sample

    all_samples_duplicated = []
    max_el_by_datatype = max(numel_by_datatype.values())

    for sample in all_samples:
        ds = sample.parent.name
        dup_factor = int(
            max_el_by_datatype / numel_by_datatype[dataset_to_datatype[ds]]
        )
        all_samples_duplicated = all_samples_duplicated + dup_factor * [sample]

    return all_samples_duplicated

