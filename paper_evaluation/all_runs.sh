#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

IS_DEBUG=0
SAVE_NAME="summary"
SAVE_FOLDER="$PWD"

# CellSAM parameters
CELLSAM_BBOX_THRESHOLD=0.4
CELLSAM_IOU_THRESHOLD=0.4
CELLSAM_MASK_THRESHOLD=0.5
CELLSAM_SAM_LOCATOR="anchor"
CELLSAM_SAM_PROMPTS="box"
CELLSAM_NUM_QUERY_POSITION=3500
CELLSAM_NUM_QUERY_PATTERN=1
CELLSAM_CKPT="$HOME/.deepcell/models/cellsam_general.pth"

#### mixed eval
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name deepbacs --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name Gendarme_BriFi --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name YeaZ --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name YeastNet --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name dsb_fixed --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name cellpose --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name H_and_E --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name ep_phase_microscopy_all --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name tissuenet_wholecell --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name omnipose --bbox_threshold $CELLSAM_BBOX_THRESHOLD --iou_threshold $CELLSAM_IOU_THRESHOLD --mask_threshold $CELLSAM_MASK_THRESHOLD --is_debug $IS_DEBUG --sam_locator $CELLSAM_SAM_LOCATOR --sam_prompts $CELLSAM_SAM_PROMPTS --model_name SAM --num_query_position $CELLSAM_NUM_QUERY_POSITION --num_query_pattern $CELLSAM_NUM_QUERY_PATTERN --ckpt $CELLSAM_CKPT

# CellPose parameters
CELLPOSE_CKPT="$HOME/.deepcell/models/cellpose_mixed_Max_0_no_CLAHE_no_pad_0shot_chan_3_chan2_2.pt"

#mixed eval
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name deepbacs --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name Gendarme_BriFi --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name YeaZ --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name YeastNet --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name dsb_fixed --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name cellpose --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name H_and_E --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name ep_phase_microscopy_all --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name tissuenet_wholecell --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
python eval_main.py --save_name $SAVE_NAME --save_folder $SAVE_FOLDER --dataset_name omnipose --model_name cellpose --is_debug $IS_DEBUG --ckpt $CELLPOSE_CKPT
