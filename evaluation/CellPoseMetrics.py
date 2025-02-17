import numpy as np
import cellpose.metrics as cp_eval

from evaluator import Evaluator


class CellPoseMetrics(Evaluator):
    def __init__(self):
        super().__init__()

    def compute_metrics(self, prediction, ground_truth, metrics = ['f1_batch_mean', 'f1_batch_std', 'f1', 'mean_ap', 'jaccard', 'mean_aji', 'dice', 'recall', 'precision']):
        # Calculate the evaluation metrics
        valid_metrics = ['f1_batch_mean', 'f1_batch_std', 'f1', 'mean_ap', 'jaccard', 'mean_aji', 'dice', 'recall', 'precision']
        # check performance using cellpose metrics
        aji = cp_eval.aggregated_jaccard_index(ground_truth, prediction)
        ap, tp_list, fp_list, fn_list = cp_eval.average_precision(ground_truth,
                                                   prediction,
                                                   threshold=[0.5])


        image_metric_dict = {}
        image_valid_metrics = ['f1', 'jaccard', 'dice', 'recall', 'precision', 'ap', 'aji', 'tp', 'fp', 'fn']
        for metric in image_valid_metrics:
            image_metric_dict[metric] = []
        image_metric_dict['ap'] = ap # shape (n_image, 1) since we only have one threshold=[0.5]
        image_metric_dict['tp'] = tp_list # shape (n_image, 1)
        image_metric_dict['fp'] = fp_list # shape (n_image, 1)
        image_metric_dict['fn'] = fn_list # shape (n_image, 1)
        image_metric_dict['aji'] = aji # shape (n_image)

        # calculate per image statistics
        for ind, (tp, fp, fn) in enumerate(zip(tp_list, fp_list, fn_list)):
            # tp, fp, fn = tp[0], fp[0], fn[0]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            # f1 = 2 * (precision * recall) / (precision + recall) # removed as it could give nan
            f1 = tp / (tp + 0.5 * (fp + fn))
            jaccard = tp / (tp + fp + fn)
            dice = 2 * tp / (2 * tp + fp + fn)
            image_metric_dict['precision'].append(precision)
            image_metric_dict['recall'].append(recall)
            image_metric_dict['f1'].append(f1)
            image_metric_dict['jaccard'].append(jaccard)
            image_metric_dict['dice'].append(dice)
        
        for metric in image_valid_metrics:
            image_metric_dict[metric] = np.asarray(image_metric_dict[metric]).astype('float64')
        self.image_metric_dict = image_metric_dict
        # print(image_metric_dict)

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> dataset evaluate result <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        tps = np.asarray(tp_list).sum(axis=0)
        fps = np.asarray(fp_list).sum(axis=0)
        fns = np.asarray(fn_list).sum(axis=0)

        # Calculate Dataset Statistics
        result_metrics = {}
        result_metrics['mean_ap'] = ap.mean(axis=0)
        result_metrics['mean_aji'] = np.asarray([aji.mean()]) # convert float to len=1 numpy array to match the format
        result_metrics['precision'] = tps / (tps + fps)
        result_metrics['recall'] = tps / (tps + fns)
        result_metrics['f1'] = tps / (tps + 0.5 * (fps + fns))
        result_metrics['jaccard'] = tps / (tps + fps + fns)
        result_metrics['dice'] = 2 * tps / (2 * tps + fps + fns)
        result_metrics['f1_batch_mean'] = np.mean(image_metric_dict['f1'], axis=0)
        result_metrics['f1_batch_std'] = np.std(image_metric_dict['f1'], axis=0)

        dataset_metric_dict = {}
        for metric in metrics:
            if metric in valid_metrics:
                dataset_metric_dict[metric] = result_metrics[metric].astype('float64')[0] # convert len=1 numpy array to float64
            else:
                raise ValueError(f"Metric {metric} not supported for CellPoseMetrics")

        if self.return_image_metrics:  #TODO: dataset_metric_dict and result_metrics are flipped
            return result_metrics, image_metric_dict, dataset_metric_dict

        else:
            return result_metrics, dataset_metric_dict


