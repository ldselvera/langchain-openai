class EvaluationMetrics:
    def __init__(self, chosen_metrics):
        self.chosen_metrics = chosen_metrics

    def calculate_metrics(self, predicted_labels, ground_truth_labels):
        """
        Calculates the specified evaluation metrics for the AutoML process.

        Parameters:
        - predicted_labels (list or ndarray): Predicted labels from the model.
        - ground_truth_labels (list or ndarray): Ground truth labels.

        Returns:
        - metrics (dict): Dictionary of calculated evaluation metrics.
        """
        metrics = {}
        for metric in self.chosen_metrics:
            if metric == 'accuracy':
                metrics[metric] = accuracy_score(ground_truth_labels, predicted_labels)
            elif metric == 'precision':
                metrics[metric] = precision_score(ground_truth_labels, predicted_labels)
            elif metric == 'recall':
                metrics[metric] = recall_score(ground_truth_labels, predicted_labels)
            elif metric == 'f1':
                metrics[metric] = f1_score(ground_truth_labels, predicted_labels)
            else:
                raise ValueError(f"Invalid evaluation metric: {metric}")

        return metrics