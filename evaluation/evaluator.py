import json


class Evaluator:
    def __init__(self, return_image_metrics=False):
        self.image_metric_dict = None
        self.dataset_metric_dict = None
        self.return_image_metrics = return_image_metrics

    def evaluate(self, predicted_masks: list, ground_truth_masks: list, metrics: list):
        if self.return_image_metrics:
            try:
                self.dataset_metric_dict, self.image_metric_dict, self.eval_metrics = self.compute_metrics(predicted_masks,
                                                                            ground_truth_masks,
                                                                            metrics,
                                                                                )
            except:
                self.dataset_metric_dict, self.eval_metrics = self.compute_metrics(predicted_masks,
                                                                                            ground_truth_masks,
                                                                                            metrics,
                                                                                            )
                self.image_metric_dict = {}
        else:
            self.dataset_metric_dict, self.eval_metrics = self.compute_metrics(predicted_masks,
                                                                    ground_truth_masks,
                                                                    metrics,
                                                                    )
        return self.eval_metrics


    def save_metrics(self, path):
        with open(path, 'w+') as fout:
            json.dump(self.dataset_metric_dict, fout)

    def compute_metrics(self, prediction, ground_truth, metric):
        raise NotImplementedError


