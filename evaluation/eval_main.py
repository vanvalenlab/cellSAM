import argparse
import os
import random
import time
from typing import Tuple, List

import fastremap
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from cellpose.utils import fill_holes_and_remove_small_masks
from joblib import Parallel, delayed
from pkg_resources import resource_filename
from scipy import ndimage
from segment_anything.utils.amg import remove_small_regions
from skimage.exposure import adjust_gamma
from tqdm import tqdm

from CellPoseMetrics import CellPoseMetrics
from mix_dataset import MixDataset
# from wsi.sam_inference import CellSAM

def extract_masks(y, return_masks=False):
    coords_per_heatmap = []
    labels_per_heatmap = []
    boxes_per_heatmap = []
    masks = []

    for i in range(y.shape[0]):
        mask_in_batch = y[i]
        mask_ids = torch.unique(mask_in_batch)
        mask_ids = mask_ids[mask_ids != 0]  # Exclude background (ID=0)

        if len(mask_ids) == 0:
            coords_per_heatmap.append(torch.empty(0, 2))
            boxes_per_heatmap.append(torch.empty(0, 4))
            labels_per_heatmap.append([])
            continue

        coords = []
        boxes = []
        labels = []

        for mask_id in mask_ids:
            mask = (mask_in_batch == mask_id).to(torch.int32)

            # Bounding box computation
            nonzero_y, nonzero_x = torch.nonzero(mask, as_tuple=True)
            y_min, y_max = nonzero_y.min(), nonzero_y.max()
            x_min, x_max = nonzero_x.min(), nonzero_x.max()
            box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

            # Center of mass calculation
            total = mask.sum().item()
            y_com = (nonzero_y.float().sum() / total).item()
            x_com = (nonzero_x.float().sum() / total).item()
            com = [x_com, y_com]

            # Append results
            coords.append(com)
            boxes.append(box)
            labels.append(mask_id.item())
            if return_masks:
                masks.append(mask.cpu().numpy())

        # Convert lists to tensors
        coords_per_heatmap.append(torch.tensor(coords, dtype=torch.float32))
        boxes_per_heatmap.append(torch.stack(boxes))
        labels_per_heatmap.append(labels)

    if return_masks:
        return coords_per_heatmap, labels_per_heatmap, boxes_per_heatmap, masks
    else:
        return coords_per_heatmap, labels_per_heatmap, boxes_per_heatmap


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
    'H_and_E': ['cpm15', 'cpm17', 'monusac', 'monuseg', 'nuinsseg', 'tnbc'],
}


def sum_single_image(t_masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Qilin: Sum the masks of a single image based on scores. The masks with higher score overwrite potential overlapping regions.
    return the summed mask and the sorted scores.
    """
    sorted_ind = np.argsort(scores)
    sorted_scores = scores[sorted_ind]  # ascending seq

    h, w = t_masks.shape[1:3]
    summed_pred = np.zeros([h, w, 1], dtype=np.int32)
    for scale, sub_ind in enumerate(sorted_ind):  # the mask with higher score overwrite potential overlapping regions
        non_zero_region = np.where(t_masks[sub_ind])
        summed_pred[non_zero_region] = int(scale + 1)

    return summed_pred, sorted_scores


def sum_masks_parallel(all_thresholded_masks: List[np.ndarray], all_scores: List[np.ndarray]) -> Tuple[
    List[np.ndarray], List[np.ndarray]]:
    """
    Qilin: parallel verion to sum the masks of all images based on scores. The masks with higher score overwrite potential overlapping regions.
    Help coco evaluation to get better results with scores. Not significant for other metrics.
    return the summed masks and the sorted scores.
    Perf gain compared to the serial version. (36.2s vs 184.977527s, 5x faster)


    Currently not used!
    """
    start_time = time.perf_counter()
    print('Start summing binary masks...')
    # use joblib to parallelize
    results = Parallel(n_jobs=-1)(delayed(sum_single_image)(t_masks, scores) for t_masks, scores in
                                  tqdm(list(zip(all_thresholded_masks, all_scores))))

    summed_preds = []
    sorted_all_scores = []
    for summed_pred, sorted_scores in results:
        summed_preds.append(summed_pred)
        sorted_all_scores.append(sorted_scores)
    print(f'Done summing masks. time: {time.perf_counter() - start_time:4f}s')
    return summed_preds, sorted_all_scores


def sum_masks(all_thresholded_masks: List[np.ndarray], all_scores: List[np.ndarray]) -> Tuple[
    List[np.ndarray], List[np.ndarray]]:
    """
    Qilin:Sum the masks of all images based on scores. The masks with higher score overwrite potential overlapping regions.
    Help coco evaluation to get better results with scores. Not significant for other metrics.
    return the summed masks and the sorted scores.

    Currently not used!
    """
    print('Start summing binary masks...')
    start_time = time.perf_counter()
    summed_preds = []
    sorted_all_scores = []
    for t_masks, scores in tqdm(list(zip(all_thresholded_masks, all_scores))):
        assert (len(t_masks) == len(scores))
        # print(t_masks.shape, scores.shape)
        sorted_ind = np.argsort(scores)
        sorted_scores = scores[sorted_ind]  # ascending seq

        h, w = t_masks.shape[1:3]
        summed_pred = np.zeros([h, w, 1], dtype=np.int32)
        for scale, sub_ind in enumerate(
                sorted_ind):  # the mask with higher score overwrite potential overlapping regions
            non_zero_region = np.where(t_masks[sub_ind])
            summed_pred[non_zero_region] = (scale + 1)

        # break

        sorted_all_scores.append(sorted_scores)
        summed_preds.append(summed_pred)
    print(f'Done summing masks. time: {time.perf_counter() - start_time:4f}s')
    return summed_preds, sorted_all_scores


def keep_largest_object(img):
    """
    Keep only the largest object in the binary image (np.array).
    """
    # Convert image to numpy array
    img_array = img

    # Label connected regions of the binary image
    label_image, num_features = ndimage.label(img_array)

    # Create a histogram of label frequency
    label_histogram = np.bincount(label_image.ravel())
    label_histogram[0] = 0  # Clear the background label

    # Find the label of the largest object
    largest_object_label = label_histogram.argmax()

    # Set all other objects to the background value
    cleaned_array = np.where(label_image == largest_object_label, img_array.max(), 0)

    # Convert the array back to an image
    # cleaned_img = Image.fromarray(cleaned_array.astype(np.uint8))

    return cleaned_array


def get_cellpose_channel(dataset: str, input_channel_type: str, output_channel_type: str, chan: int = 0,
                         chan2: int = 0) -> List[int]:
    # Qilin: Define channels for cellpose model to use for evaluation
    if dataset == "tissuenet":
        if output_channel_type not in ["whole_cell", "nuclear"]:
            raise ValueError(
                "invalid output_channel_type for cellpose model on tissuenet - please select from 'whole_cell', 'nuclear'")
        if input_channel_type == "both":
            if output_channel_type == "nuclear":
                channels = [2, 3]
            elif output_channel_type == "whole_cell":
                channels = [3, 2]
        elif input_channel_type == "nuclear":
            channels = [2, 0]
        elif input_channel_type == "whole_cell":
            channels = [3, 0]
        else:
            raise ValueError(
                "invalid input_channel_type for cellpose model on tissuenet - please select from 'whole_cell', 'nuclear', or 'both'")
    elif dataset == "cellpose":
        channels = [2, 1]
    elif dataset == "omnipose":
        channels = [0, 0]
    else:
        # raise ValueError(f'Invalid dataset {dataset} and output_channel_type {output_channel_type} combination for cellpose model!')
        print(f'user defined dataset, use user defined [{chan},{chan2}] to set channels')
        channels = [chan, chan2]
    return channels

def parse_args():
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--nshots", type=int, default=50)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--pretrain_dir", type=str, default="")
    parser.add_argument("--eval_dir", type=str, default="./results/evals")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--project", type=str, default="sam_new_eval_oct24_2024")
    parser.add_argument("--group", type=str, default="")

    # model - cellpose args
    parser.add_argument("--cellpose_model_type", type=str, default="cyto3")
    parser.add_argument("--chan", type=int, default=3)
    parser.add_argument("--chan2", type=int, default=2)
    parser.add_argument("--with_size", type=int, default=0, help="0 or 1, whether the pretrained_model has size model.")

    # model - sam args
    parser.add_argument("--sam_locator", type=str, default="ground_truth")  # options are ground_truth, cellfinder
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--sam_prompts", nargs="+", default=["box"])

    # data args
    parser.add_argument("--dataloader_root", type=str, default="data/dataset/")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--input_channel_type", type=str, default="")
    parser.add_argument("--output_channel_type", type=str, default="")
    parser.add_argument("--normalize", type=int, default=-1,
                        help="0 or 1, whether to normalize the input image to [0, 1]")
    parser.add_argument("--CLAHE", type=int, default=0,
                        help="0 or 1, whether to use CLAHE to enhance the contrast of the image")

    # eval args
    parser.add_argument("--config_file", type=str, default="./eval_cfg.yaml")
    # metric_type options are ["coco", "cellpose", "deepcell", "stardist"]
    parser.add_argument("--metric_type", nargs="+", default=["coco", "cellpose", "deepcell", "stardist"])
    parser.add_argument('--max_plots', type=int, default=20)

    # general args
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--is_debug", type=int, default=0)

    parser.add_argument("--num_query_position", type=int, default=300)
    parser.add_argument("--num_query_pattern", type=int, default=3)
    parser.add_argument("--use_vanilla_sam_seg", type=int, default=0)
    parser.add_argument("--bbox_threshold", type=float, default=0.4)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.45)
    parser.add_argument("--spatial_prior", type=str, default="learned")
    parser.add_argument("--crop_size", type=int, default=0)  # 0 for orig image, 512 for crop_512 dataset
    parser.add_argument("--image_size", type=str, default="crop_512")  # Not used now. choices 'crop_512', 'orig'
    # post sam nms args
    parser.add_argument("--use_nms", type=int, default=0)
    parser.add_argument("--sum_on_scores", type=int, default=0,
                        help="0(default) or 1, whether to sum masks based on scores")
    parser.add_argument("--nms_iou_threshold", type=float, default=0.7)  #
    parser.add_argument("--use_cellpose_filter", type=int, default=0)
    parser.add_argument("--extra_normalize", type=int, default=0)  # for livecell
    parser.add_argument("--auto_sam_area_threshold", type=int, default=1000)

    parser.add_argument("--print_imgs_with_metrics", type=int, default=0)
    parser.add_argument("--additional_postprocessing", type=int, default=0)

    # wsi args
    parser.add_argument("--lower_contrast_threshold", type=float, default=0.025)
    parser.add_argument("--upper_contrast_threshold", type=float, default=0.1)

    parser.add_argument("--medium_cell_threshold", type=float, default=0.002)
    parser.add_argument("--large_cell_threshold", type=float, default=0.015)
    parser.add_argument("--medium_cell_max", type=int, default=60)
    parser.add_argument("--medium_mean_diff_threshold", type=float, default=0.1)
    parser.add_argument("--cells_min_size", type=int, default=500)
    parser.add_argument("--border_size", type=int, default=5)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--iou_depth", type=int, default=100)

    parser.add_argument("--SAM_ft_ckpt_path", type=str, default="")
    parser.add_argument("--dice_loss", type=str, default="dice_only")

    parser.add_argument("--save_folder", type=str, default="results/evals")
    parser.add_argument("--save_name", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":

    # seed everything for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    args = parse_args()

    # Load configuration file
    cfg = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)

    # Update cfg only with attributes from args that are not empty strings and not -1
    cfg.update({k: v for k, v in vars(args).items() if v not in ["", -1]})
    # hard coded stuff for eval
    cfg['unfrozen_sam_parts'] = ''
    cfg['freeze_cellfinder'] = True

    print('#################')
    print(f'loading: {args.dataset_name}')
    print('#################')

    ds = args.dataset_name
    if ds in ['deepbacs', 'omnipose', 'ep_phase_microscopy_all', 'H_and_E']:
        ds = DATAMAP[ds]

    dataset = MixDataset(
        split='test',
        dataset=ds,
        crop_size=args.crop_size,
        root_dir=args.dataloader_root,
        data_type='npy',
        normalize=args.normalize,
        CLAHE=args.CLAHE,
        duplicate=False
    )

    # Set run_name if it's empty and assign based on model-specific logic
    if not args.run_name:
        if args.model_name == "SAM":
            if args.ckpt and 'mixed' not in args.ckpt:
                run_name = f'FT_{args.model_name}_{args.sam_locator}_{args.sam_prompts[0]}Prompt_{args.dataset_name}_train_{args.dataset_name}_eval'
            elif args.ckpt and 'mixed' in args.ckpt:
                run_name = f'FT_{args.model_name}_{args.sam_locator}_{args.sam_prompts[0]}Prompt_MixedData_train_{args.dataset_name}_eval'
            else:
                run_name = f'ZS_{args.model_name}_{args.sam_locator}_{args.sam_prompts[0]}Prompt_{args.dataset_name}'
        elif args.model_name == "mesmer":
            run_name = f'mesmer_{args.dataset_name}'
        elif args.model_name == "cellpose":
            run_name = f'cellpose_{args.dataset_name}'  # Assuming default for cellpose if needed

    # convert images and masks to list of torch tensors
    data = [(d[0], d[1]) for d in dataset]
    imgs, masks = zip(*data)
    paths = [d[2] for d in dataset]

    if not os.path.exists('results/evals'):
        os.makedirs('results/evals')

    if args.dataset_name == "LIVECell" or args.dataset_name == "LIVECell_good":
        metadata = []
        for idx in range(len(dataset)):
            meta = dataset.get_metadata(idx)
            meta = str(meta)
            meta = meta.split('/')[-1]
            meta = meta.split('_')[-6]
            metadata.append(meta)

        # save metadata
        import pandas as pd

        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv('results/evals/metadata_LIVECELLgood_{crop_size}_{image_size}.csv'.format(crop_size=args.crop_size,
                                                                                             image_size=args.image_size),
                       index=False, header=False)

        # unique items in metadata
        metadata_unique = ['A172', 'SKOV3', 'SHSY5Y', 'SkBr3', 'BT474', 'BV2', 'Huh7', 'MCF7']
        chose_type = metadata_unique[0]
        # get indices of the chosen type
        chose_type_idx = [i for i, x in enumerate(metadata) if x == chose_type]
        chose_type_idx = np.array(chose_type_idx)
        print(chose_type)

    if args.dataset_name == "tissuenet_wholecell":
        metadata = []
        for idx in range(len(dataset)):
            meta = dataset.get_metadata(idx)
            meta = str(meta)
            meta = meta.split('_')[-1]
            metadata.append(meta)
        import pandas as pd

        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(
            'results/evals/metadata_tissuenet_wholecell_{crop_size}_{image_size}.csv'.format(crop_size=args.crop_size,
                                                                                             image_size=args.image_size),
            index=False, header=False)

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    start_time = time.time()

    # load model
    if args.model_name == "cellpose":
        from models import CellPoseModel

        # update cfg pretrain: 0 for cellpose official model, 1 for user fine-tuned model
        cfg['model']['pretrain'] = args.pretrain

        # update cfg model_type
        if args.cellpose_model_type == "":
            cellpose_model_type = cfg["cellpose"]["model_type"]
        else:
            cellpose_model_type = args.cellpose_model_type
            cfg["cellpose"]["model_type"] = cellpose_model_type

        # get channels  
        if args.chan == -1:
            chan = cfg["cellpose"]["chan"]
        else:
            chan = args.chan
            cfg["cellpose"]["chan"] = chan

        if args.chan2 == -1:
            chan2 = cfg["cellpose"]["chan2"]
        else:
            chan2 = args.chan2
            cfg["cellpose"]["chan2"] = chan2
        
        # get with_size
        if args.with_size not in [0, 1]:
            with_size = cfg["cellpose"]["with_size"]
        else:
            with_size = args.with_size
            cfg["cellpose"]["with_size"] = with_size
        

        # load built-in model if model_type is valid
        if not args.pretrain:
            MODEL_NAMES = ['cyto', 'nuclei', 'cyto2', 'livecell', 'tissuenet', 'cyto3', 'tissuenet_cp3',
                           'livecell_cp3', 'yeast_PhC_cp3', 'yeast_BF_cp3', 'bact_phase_cp3', 'bact_fluor_cp3',
                           'deepbacs_cp3', 'cyto2_cp3']
            assert args.cellpose_model_type in MODEL_NAMES, f"invalid model_type {args.cellpose_model_type} for built-in model, - please select from {MODEL_NAMES}."
            # define wandb run name
            run_name = args.run_name
            save_name = run_name
            cfg['cellpose']['pretrained_model'] = ""
        else:
            # load pretrained fine-tuned model
            # define wandb run name and group
            assert args.nshots == 0, f'Self-fine-tuning cellpose model only supports [full] for nshots. Got nshots: {args.nshots}.'
            run_name = args.run_name
            save_name = run_name
            if args.ckpt == "":
                raise ValueError('ckpt path is empty for cellpose fine-tuned model!')
            else:  # load cellpose model directly from ckpt
                cfg['cellpose']['pretrained_model'] = args.ckpt

        app = CellPoseModel(cfg, None)

        # Update cfg channels
        chan = app.channels[0]
        chan2 = app.channels[1]
        args.chan = chan
        args.chan2 = chan2
        cfg["cellpose"]["chan"] = chan
        cfg["cellpose"]["chan2"] = chan2

    elif args.model_name == "mesmer":
        from deepcell.applications import Mesmer

        if args.pretrain:
            import tensorflow as tf

            mesmer_pretrain_path = os.path.join(args.pretrain_dir,
                                                f"mesmer/{args.dataset_name}/{args.output_channel_type}")
            save_name = f"mesmer_{args.dataset_name}_{args.output_channel_type}_{args.nshots}shot"
            model_path = os.path.join(mesmer_pretrain_path, save_name)
            print(f'loading pretrained memesr model - {model_path}')
            model = tf.keras.models.load_model(model_path)
            app = Mesmer(model=model)

        else:
            print('loadding from mesmer application')
            save_name = f"mesmer_app_{args.dataset_name}"
            app = Mesmer()

    elif args.model_name == "SAM":
        from models.sam import SAM

        save_name = args.run_name
        if args.run_name == "":
            run_name = f'sam_{args.dataset_name}_{args.output_channel_type}_{args.nshots}shot_{args.sam_locator}_locator_{current_time}'
            save_name = f"sam_{args.dataset_name}_{args.output_channel_type}_{args.nshots}shot_chan_{args.chan}_chan2_{args.chan2}.pt"

        app = SAM(cfg, None)
        app.target_image_size = cfg["data"]["img_size"]
        if not args.ckpt == "":
            state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
            try:
                # filter out the backbone weights
                app.load_state_dict(state_dict, strict=True)
                print('warning, loading ckpt not strictly: reverse me later')
                print('loaded ckpt')
            except Exception as e:
                print(e)
                check = True
                if 'pos_emb' in str(e) and not 'image_encoder' in str(e):
                    app.load_state_dict(state_dict, strict=False)
                    print('loaded ckpt!!!')
                    check = False
                if check:
                    if 'state_dict' in state_dict.keys():
                        state_dict = state_dict['state_dict']
                        ret = None
                        try:
                            ret = app.load_state_dict(state_dict)
                        except Exception as e:
                            print(e)
                            app.load_state_dict(state_dict, strict=False)
                        if ret:
                            print(ret)
                        print('loaded ckpt!!!')
                    elif 'model' in state_dict.keys():
                        state_dict = state_dict['model']
                        backbone_state_dict = {}
                        detector_state_dict = {}
                        for k, v in state_dict.items():
                            if 'backbone' in k:
                                # remove backbone. from key
                                k = k[9:]
                                # remove body. from key
                                k = k[5:]
                                backbone_state_dict[k] = v
                            else:
                                pass
                        app.model.image_encoder.load_state_dict(backbone_state_dict, strict=False)
                        # if app has attribute cellfinder
                        if hasattr(app, 'cellfinder'):
                            app.cellfinder.decode_head.load_state_dict(state_dict)
                        print('loaded ckpt!!!')
                    else:
                        raise ValueError('ckpt not recognized')

        if args.SAM_ft_ckpt_path != "":
            state_dict = torch.load(args.SAM_ft_ckpt_path, map_location=torch.device('cpu'))
            # test if state_dict is a dict
            if isinstance(state_dict, dict):
                state_dict = state_dict['state_dict']
                # only take parts of the state_dict that start with model
                state_dict = {k: v for k, v in state_dict.items() if 'model.' in k}
            else:
                app.model_cp = state_dict.model
            print('loaded tuned ckpt')

        app = app.eval()
        app = app.cuda()

    elif args.model_name == "SAM_wsi":
        def get_local_model(model_path: str):
            """
            Returns a loaded CellSAM model from a local path.
            """
            config_path = resource_filename(__name__, './wsi/modelconfig.yaml')
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)

            model = CellSAM(config)
            model.load_state_dict(torch.load(model_path), strict=False)
            return model


        app = get_local_model(model_path=args.ckpt)
        app.bbox_threshold = args.bbox_threshold
        app = app.eval()
        app = app.cuda()
        # model.iou_threshold = 0.9
    else:
        raise ValueError(f'{args.model_name} model not recognized')


    def normalize_images(imgs):
        """
        Normalize each channel of all images in a list separately.

        Args:
            imgs (list of torch.Tensor): List of images, each with 3 channels (C, H, W).

        Returns:
            list of torch.Tensor: List of normalized images.
        """

        def normalize_channel(image):
            # Normalize each channel separately
            normalized_channels = [
                (channel - channel.min()) / (channel.max() - channel.min() + 1e-7)
                # Add epsilon to avoid division by zero
                for channel in image
            ]
            return torch.stack(normalized_channels, dim=0)

        return [normalize_channel(image) for image in imgs]


    if args.is_debug:
        im_list = range(1200,1500)
        imgs = [imgs[i] for i in im_list]
        masks = [masks[i] for i in im_list]

        plt.imshow(imgs[0][2])
        plt.show()

    # normalize images (channel wise)
    imgs = normalize_images(imgs)

    if args.extra_normalize:
        import numpy as np

        import numpy as np
        import matplotlib.pyplot as plt

        from skimage.filters import rank

        # Load an example image
        img = imgs[0].numpy()
        # move 1st channel to last
        img = img[2]

        import numpy as np
        import matplotlib.pyplot as plt

        imgs = [img.numpy() for img in imgs]
        # only last channel
        imgs = [img[2] for img in imgs]

        # import disk
        from skimage.morphology import disk

        # imgs = [exposure.equalize_hist(img) for img in imgs]
        footprint = disk(30)
        imgs = [rank.equalize(img, footprint=footprint) for img in tqdm(imgs)]

        # adjust gamma
        imgs = [adjust_gamma(img, gamma=3.0) for img in imgs]

        # back to tensor
        imgs = [torch.tensor(img) for img in imgs]

        # add channel dim
        imgs = [img.unsqueeze(0) for img in imgs]

        imgs = [img.repeat(3, 1, 1) for img in imgs]
        # make first 2 channels 0s

        for im in imgs:
            im[0, :, :] = 0
            im[1, :, :] = 0

    print(f'starting predictions for {args.run_name}')
    global_threshold = 0.2
    start_time = time.time()
    if "SAM" in args.model_name:
        # TODO: do batching in predict
        all_predictions = []
        all_scores = []
        all_thresholded_masks = []
        batch_size = 1
        for i in tqdm(range(0, len(imgs), batch_size)):
            batch = imgs[i:i + batch_size]
            masks_batch = masks[i:i + batch_size]
            coords_per_heatmap = None
            boxes_per_heatmap = None
            if args.sam_locator == "ground_truth":
                coords_per_heatmap = []
                labels_per_heatmap = []
                boxes_per_heatmap = []
                for mask_in_batch in masks_batch:
                    mask_ids = np.unique(mask_in_batch[0])
                    coords = []
                    boxes = []
                    for mask_id in mask_ids:
                        if mask_id == 0:
                            continue
                        mask_id = int(mask_id)
                        mask = mask_in_batch == mask_id
                        _mask = np.zeros_like(mask)
                        _mask[mask > 0] = 1
                        mask = _mask
                        com = ndimage.center_of_mass(mask[0])
                        nonzero_elements = np.nonzero(mask)
                        y_min = np.min(nonzero_elements[1])
                        y_max = np.max(nonzero_elements[1])
                        x_min = np.min(nonzero_elements[2])
                        x_max = np.max(nonzero_elements[2])

                        box = [x_min, y_min, x_max, y_max]
                        boxes.append(box)
                        coords.append(com)

                    coords = torch.from_numpy(np.array(coords)).cuda()
                    boxes = torch.from_numpy(np.array(boxes)).cuda()
                    # flip coords
                    coords = torch.flip(coords, dims=[1])
                    coords_per_heatmap.append(coords)
                    boxes_per_heatmap.append(boxes)
                coords_per_heatmap = coords.unsqueeze(0)
                boxes_per_heatmap = boxes.unsqueeze(0)
                # integrate in model
                app.img_scale_factor = 2
                app.mask_threshold = 0.5
            elif args.sam_locator == "anchor" or args.sam_locator == "sam_auto":
                coords_per_heatmap = None
                app.img_scale_factor = 4
                app.threshold = args.iou_threshold
                app.bbox_threshold = args.bbox_threshold
            else:
                raise ValueError('locator not recognized')

            app.threshold = args.iou_threshold
            app.bbox_threshold = args.bbox_threshold
            cfg["iou_threshold"] = args.iou_threshold
            cfg["mask_threshold"] = args.mask_threshold
            app.iou_threshold = args.iou_threshold
            app.mask_threshold = args.mask_threshold

            if args.model_name == "SAM":
                segmentation_predictions, thresholded_masks, low_masks, scores = app.predict(batch,
                                                                                             coords_per_heatmap=coords_per_heatmap,
                                                                                             boxes_per_heatmap=boxes_per_heatmap,
                                                                                             global_threshold=global_threshold,
                                                                                             gt_map=masks_batch,
                                                                                             prompts=args.sam_prompts,
                                                                                             return_lower_level_comps=True,
                                                                                             img_num=i,
                                                                                             )
            else:
                raise ValueError('model name not recognized')

            if args.use_cellpose_filter:

                thresholded_masks_summed = (
                        thresholded_masks
                        * np.arange(1, thresholded_masks.shape[0] + 1)[:, None, None]
                )
                segmentation_predictions = np.max(thresholded_masks_summed, axis=0)

                segmentation_predictions = fill_holes_and_remove_small_masks(segmentation_predictions, min_size=25)

                from skimage.morphology import disk
                from skimage.morphology import binary_opening, binary_closing, binary_erosion, binary_dilation

                # compute convex hull for each mask
                mask_values = np.unique(segmentation_predictions)
                new_masks = []
                for mask_id, mask_value in enumerate(mask_values):
                    if mask_value == 0:
                        continue
                    mask = segmentation_predictions == mask_value

                    mask, changed = remove_small_regions(mask, 40, mode="holes")
                    mask, changed = remove_small_regions(mask, 40, mode="islands")

                    # # Define the structuring element for morphological operations
                    selem = disk(2)
                    # Apply morphological opening
                    opened_mask = binary_opening(mask, selem)
                    # Apply morphological closing
                    closed_mask = binary_closing(opened_mask, selem)
                    mask = closed_mask

                    selem = disk(10)
                    # dilation and erosion
                    mask = binary_dilation(mask, selem)
                    mask = binary_erosion(mask, selem)

                    # threshold
                    mask = mask > 0.5

                    mask = mask.astype(np.uint8) * mask_value
                    new_masks.append(mask)
                segmentation_predictions = np.max(new_masks, axis=0)

            all_predictions.append(segmentation_predictions)
            all_scores.append(scores)
            all_thresholded_masks.append(thresholded_masks)
        preds = all_predictions
        preds = [np.expand_dims(pred, axis=-1) for pred in preds]
        all_thresholded_masks = [np.expand_dims(mask, axis=-1) for mask in all_thresholded_masks]

        if cfg['sum_on_scores']:
            summed_preds, sorted_all_scores = sum_masks_parallel(all_thresholded_masks, all_scores)
            if args.use_cellpose_filter:  # CANNOT use scores anymore!
                summed_preds_filled = [
                    fill_holes_and_remove_small_masks(summed_pred[:, :, 0], min_size=25)[:, :, np.newaxis] for
                    summed_pred in summed_preds]
                preds = summed_preds_filled
            else:
                preds = summed_preds

    elif args.model_name == "mesmer":
        all_predictions = []
        for i in range(len(imgs)):
            im = imgs[i].unsqueeze(-1).permute(3, 1, 2, 0)[..., 1:]
            if args.dataset_name in ['Gendarme_BriFi', 'nuc_seg_dsb']:
                segmentation_predictions = app.predict(im, image_mpp=0.5, compartment='nuclear')
            else:
                segmentation_predictions = app.predict(im, image_mpp=0.5, compartment='whole-cell')
            all_predictions.append(np.squeeze(segmentation_predictions, 0))
            # masks_list.append(masks[i])
        segmentation_predictions = np.stack(all_predictions, axis=0)
        preds = [np.array(pred) for pred in segmentation_predictions]
    elif args.model_name == "cellpose":
        img_list = [img.permute(1, 2, 0).numpy() for img in imgs]
        preds = app.predict(img_list)

    masks = [mask.unsqueeze(0) for mask in masks]
    masks = [mask.permute(1, 2, 0).numpy() for mask in masks]
    masks = [fastremap.renumber(label, in_place=True)[0].astype(np.int32) for label in masks]
    preds = [fastremap.renumber(label, in_place=True)[0].astype(np.int32) for label in preds]

    ### Assertions
    assert type(masks) == list, f'type(masks): {type(masks)} != list'
    assert type(preds) == list, f'type(preds): {type(preds)} != list'
    assert len(masks) == len(preds), f'len(masks): {len(masks)} != len(preds): {len(preds)}'
    assert type(masks[0]) == np.ndarray, f'type(masks[0]): {type(masks[0])} != np.ndarray'
    assert type(preds[0]) == np.ndarray, f'type(preds[0]): {type(preds[0])} != np.ndarray'
    assert masks[0].shape == preds[0].shape, f'masks[0].shape: {masks[0].shape} != preds[0].shape: {preds[0].shape}'

    assert preds[0].dtype == (np.int32 or np.int64), f'preds[0].dtype: {preds[0].dtype} != (np.int32 or np.int64)'
    assert masks[0].dtype == (np.int32 or np.int64), f'masks[0].dtype: {masks[0].dtype} != (np.int32 or np.int64)'
    assert masks[0].dtype == preds[0].dtype, f'masks[0].dtype: {masks[0].dtype} != preds[0].dtype: {preds[0].dtype}'
    print('finished predictions')
    print(f'time elapsed: {time.time() - start_time:4f}s')
    print()

    verbose_save = True
    save_folder = f'./results/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    batch_size = 1
    deepcell_batch_f1_scores = []
    cellpose_batch_f1_scores = []
    precisions = []
    recalls = []
    nan_list = []
    for i in range(0, len(imgs), batch_size):
        # get batch
        batch_imgs = imgs[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        batch_preds = preds[i:i + batch_size]

        cellpose_evaluator = CellPoseMetrics()

        cellpose_evaluator.evaluate(batch_preds, batch_masks, ['f1', 'precision', 'recall'])
        cellpose_batch_f1 = cellpose_evaluator.dataset_metric_dict['f1']
        if np.isnan(cellpose_batch_f1):
            print('nan detected in cellpose_batch {}'.format(i))
            cellpose_batch_f1_scores.append(0)
            precisions.append(0)
            recalls.append(0)
        else:
            cellpose_batch_f1_scores.append(cellpose_batch_f1)
            precisions.append(cellpose_evaluator.dataset_metric_dict['precision'])
            recalls.append(cellpose_evaluator.dataset_metric_dict['recall'])

    cellpose_batch_f1_mean = np.mean(cellpose_batch_f1_scores)
    cellpose_batch_f1_std = np.std(cellpose_batch_f1_scores)

    print('cellpose_batch_f1_mean')
    print(cellpose_batch_f1_mean)
    print('cellpose_batch_f1_std')
    print(cellpose_batch_f1_std)

    cellpose_batch_precision_mean = np.mean(precisions)
    cellpose_batch_recall_mean = np.mean(recalls)

    print('cellpose_batch_precision_mean')
    print(cellpose_batch_precision_mean)

    print('cellpose_batch_recall_mean')
    print(cellpose_batch_recall_mean)

