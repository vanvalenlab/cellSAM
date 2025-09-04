import argparse
import random
import fastremap
import numpy as np
import torch
from tqdm import tqdm
from mix_dataset import MixDataset
from cpm import CellPoseMetrics
import os

from models import CellPoseModel
from cellSAM.sam_inference import CellSAM

DATAMAP = {
    'deepbacs': ['2b_brightfield_dataset',
                 '2b_fluorescence_dataset',
                 '2c_e_coli', '2d_1_SplineDist_dataset',
                 '2d_2_b_subtilis', '2e_e_coli',
                 's2_stardist'],
    'omnipose': ['bact_fluor', 'bact_phase'],
    'ep_phase_microscopy_all': ['3T3_ep_microscopy',
                                'HEK293_ep_microscopy',
                                'PC3_ep_microscopy',
                                'HeLa_ep_microscopy',
                                'RAW264_ep_microscopy',
                                'A549_ep_microscopy',
                                'CHO_ep_microscopy',
                                'HeLa-S3_ep_microscopy'],
    'H_and_E': ['cpm15', 'cpm17', 'monusac', 'monuseg', 'nuinsseg', 'tnbc', 'kumar'],
    'cellpose': ['cellpose']
}


def normalize_images(imgs):
    def normalize_channel(image):
        normalized_channels = [
            (channel - channel.min()) / (channel.max() - channel.min() + 1e-7)
            for channel in image
        ]
        return torch.stack(normalized_channels, dim=0)
    return [normalize_channel(image) for image in imgs]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--save_folder", type=str, default="results/evals")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--bbox_threshold", type=float, default=0.4)
    parser.add_argument("--iou_threshold", type=float, default=0.4)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--is_debug", type=int, default=0)
    parser.add_argument("--sam_locator", type=str, default="anchor")
    parser.add_argument("--sam_prompts", nargs="+", default=["box"])
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--num_query_position", type=int, default=3500)
    parser.add_argument("--num_query_pattern", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--dataloader_root", type=str, default="")
    parser.add_argument("--cellpose_model_type", type=str, default="", help="cellpose built-in model type, e.g. cyto, cyto2, cyto3, etc.")

    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    args = parse_args()

    cfg = {
        'unfrozen_sam_parts': '',
        'freeze_cellfinder': True,
        'num_query_position': args.num_query_position,
        'num_query_pattern': args.num_query_pattern,
        'spatial_prior': 'learned',
        'data': {'img_size': 'orig'},
        'dataloader_root': None,
        'cellpose': {
            'model_type': 'cyto3',
            'chan': 3,
            'chan2': 2,
            'pretrained_model': ''
        },
        'sum_on_scores': False,
        'sam_locator': args.sam_locator,
        'mask_threshold': args.mask_threshold,
        'enc_layers': 6,
        'dec_layers': 6,
        'dim_feedforward': 1024,
        'hidden_dim': 256,
        'dropout': 0.0,
        'nheads': 8,
        'attention_type': 'RCDA',
        'num_feature_levels': 1,
        'device': 'cuda',
        'seed': 42,
        'num_classes': 2,
        'model': {'pretrain': 0}
    }

    cfg.update({k: v for k, v in vars(args).items() if v not in ["", -1]})

    ds = args.dataset_name
    if ds in DATAMAP:
        ds = DATAMAP[ds]

    dataset = MixDataset(
        split='test',
        dataset=ds,
        crop_size=0,
        root_dir=cfg["dataloader_root"],
        data_type='npy',
        normalize=False,
        CLAHE=False,
        duplicate=False
    )

    data = [(d[0], d[1]) for d in dataset]
    imgs, masks = zip(*data)
    if args.model_name == "cellpose":
        # update cfg model_type for cellpose built-in models
        if not args.cellpose_model_type == "":

            cfg["cellpose"]["model_type"] = args.cellpose_model_type
        
        # use custom checkpoint if provided
        if args.ckpt:
            cfg['cellpose']['pretrained_model'] = args.ckpt
            cfg['model']['pretrain'] = 1
            cfg["cellpose"]["with_size"] = 1
        app = CellPoseModel(cfg)

    elif args.model_name == "SAM":
        app = CellSAM(cfg)
        app.target_image_size = "crop_512" # TODO: seems to be unused
        app.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')), strict=False)
        app = app.eval().cuda()
    else:
        raise ValueError(f'{args.model_name} model not recognized')

    # Set parameters (same as original)
    if "SAM" in args.model_name:
        app.img_scale_factor = 4  # From original eval_main_old.py
        app.threshold = args.iou_threshold
        app.bbox_threshold = args.bbox_threshold
        app.iou_threshold = args.iou_threshold
        app.mask_threshold = args.mask_threshold

    if args.is_debug:
        imgs = imgs[:2]
        masks = masks[:2]

    imgs = normalize_images(imgs)

    if "SAM" in args.model_name:
        preds = []

        for img in tqdm(imgs):
            result, _, _, _ = app.predict(
                [img],
                coords_per_heatmap=None,
                boxes_per_heatmap=None,
            )

            if result is None:
                segmentation_predictions = np.zeros_like(img[0].cpu().numpy(), dtype=np.int32)
            else:
                segmentation_predictions = result[0] if isinstance(result, tuple) else result
            preds.append(segmentation_predictions)

    elif args.model_name == "cellpose":
        preds = []
        img_list = [img.permute(1, 2, 0).numpy() for img in imgs]
        for im in tqdm(img_list):
            pred = app.predict(im)
            preds.append(pred)

    masks = [mask.permute(1, 2, 0).numpy().squeeze() for mask in masks]
    masks = [fastremap.renumber(label, in_place=True)[0].astype(np.int32) for label in masks]
    preds = [fastremap.renumber(np.squeeze(label), in_place=True)[0].astype(np.int32) for label in preds]

    recalls = []
    f1s = []
    for i in range(len(imgs)):
        evaluator = CellPoseMetrics()
        evaluator.evaluate([preds[i]], [masks[i]], ['f1', 'recall'])
        f1 = evaluator.dataset_metric_dict['f1']
        recall = evaluator.dataset_metric_dict['recall']
        f1s.append(0 if np.isnan(f1) else f1)
        recalls.append(0 if np.isnan(recall) else recall)

    cellpose_batch_f1_mean = np.mean(f1s)
    print(f'cellpose_batch_f1_mean: {cellpose_batch_f1_mean}')

    cellpose_batch_recall = np.mean(recalls)
    print(f'cellpose_batch_recall: {cellpose_batch_recall}')

    # Save F1 mean to CSV file
    if args.save_name:
        import csv
        os.makedirs(args.save_folder, exist_ok=True)
        csv_path = os.path.join(args.save_folder, f"{args.save_name}.csv")
        
        # Check if file exists and has header
        file_exists = os.path.exists(csv_path)
        has_header = False
        header = ['dataset', 'model', 'f1_mean']
        
        if file_exists:
            # Check if file has the header
            with open(csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                try:
                    first_row = next(reader)
                    has_header = (first_row == header)
                except StopIteration:
                    # File is empty
                    has_header = False

        # Write to CSV file
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Only write header if file doesn't exist or doesn't have header
            if not file_exists or not has_header:
                writer.writerow(header)
            writer.writerow([args.dataset_name, args.model_name, cellpose_batch_f1_mean])
            csvfile.flush()

        print(f'Results saved to: {csv_path}')
