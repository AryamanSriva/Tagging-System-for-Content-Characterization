"""
Evaluation utilities for entity extraction performance.
"""


class Evaluator:
    """Handles evaluation metrics for entity extraction."""
    
    @staticmethod
    def calculate_entity_metrics(actual, predicted):
        """
        Calculate precision, recall, and F1 score for entity extraction.
        
        Args:
            actual (dict): Dictionary of actual entities by label
            predicted (dict): Dictionary of predicted entities by label
            
        Returns:
            tuple: (precision, recall, f1)
        """
        precision, recall, f1 = 0, 0, 0
        num_entities = len(actual)

        for ent_type in actual:
            actual_entities = set(actual.get(ent_type, []))
            predicted_entities = set(predicted.get(ent_type, []))

            true_positives = len(actual_entities & predicted_entities)
            false_positives = len(predicted_entities - actual_entities)
            false_negatives = len(actual_entities - predicted_entities)

            precision_part = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall_part = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
            
            precision += precision_part
            recall += recall_part

        # Handle the case where there are no entities
        if num_entities > 0:
            precision /= num_entities
            recall /= num_entities
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f1 = 0, 0, 0

        return precision, recall, f1
    
    @staticmethod
    def evaluate_dataset(df, actual_col='Actual_Entities', predicted_col='Refined_Entities'):
        """
        Evaluate entity extraction performance on a dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame containing actual and predicted entities
            actual_col (str): Column name for actual entities
            predicted_col (str): Column name for predicted entities
            
        Returns:
            dict: Dictionary containing average metrics
        """
        metrics = df.apply(
            lambda row: Evaluator.calculate_entity_metrics(
                row[actual_col], row[predicted_col]
            ), axis=1
        )
        
        precisions, recalls, f1s = zip(*metrics)
        
        return {
            'average_precision': sum(precisions) / len(precisions),
            'average_recall': sum(recalls) / len(recalls),
            'average_f1': sum(f1s) / len(f1s)
        }
    
    @staticmethod
    def print_evaluation_results(metrics):
        """Print evaluation results in a formatted way."""
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"Average Recall: {metrics['average_recall']:.4f}")
        print(f"Average F1 Score: {metrics['average_f1']:.4f}")