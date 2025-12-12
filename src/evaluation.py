import numpy as np
from evaluate import load
from transformers import EvalPrediction
from config import Config
from typing import Dict
import torch

# --- 1. Load the Task-Specific Metric ---

# The Hugging Face `evaluate` library automatically determines the 
# correct metric(s) (e.g., accuracy, F1, MCC) based on the GLUE task name.
# MNLI uses 'accuracy'. MRPC uses 'accuracy' and 'f1'. CoLA uses 'matthews_correlation'.
metric = load("glue", Config.DATA.TASK_NAME)

# --- 2. The Core Metric Computation Function ---

def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Computes the evaluation metrics for the given predictions and labels.
    
    This function is designed to be passed directly to the Hugging Face Trainer.

    Args:
        eval_preds (EvalPrediction): A named tuple containing:
            - predictions (np.ndarray): The raw logits output by the model.
            - label_ids (np.ndarray): The ground-truth integer labels.

    Returns:
        Dict[str, float]: A dictionary mapping metric names (e.g., 'accuracy')
                          to their computed float values.
    """
    # 1. Unpack predictions and labels
    logits, labels = eval_preds
    
    # 2. Convert logits to hard predictions (the final class index)
    # The classification prediction is the index with the maximum logit value.
    predictions = np.argmax(logits, axis=-1)

    # 3. Compute the task-specific metric(s)
    # The `metric.compute()` function handles the task-specific logic.
    # It expects predictions and references (labels).
    results = metric.compute(predictions=predictions, references=labels)
    
    # For tasks like MNLI that have two validation sets ('mnli' and 'mnli-mm'), 
    # the results might need slight adjustment, but the core logic remains the same.
    # The result is already a dictionary, e.g., {'accuracy': 0.85}.
    
    return results

# --- 3. Utility Function (Optional, for easy command-line evaluation) ---

def run_standalone_evaluation(model, eval_dataloader) -> Dict[str, float]:
    """
    Runs a simple evaluation loop for a model outside of the Hugging Face Trainer.
    (This is less common but useful for checking models quickly).
    
    Args:
        model (torch.nn.Module): The student model to evaluate.
        eval_dataloader (torch.utils.data.DataLoader): The evaluation data loader.
        
    Returns:
        Dict[str, float]: The computed metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move data to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Collect predictions and labels
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # Compute final metrics
    return metric.compute(predictions=np.array(all_predictions), references=np.array(all_labels))

if __name__ == '__main__':
    # Conceptual check on the metric loading
    
    print(f"GLUE Task: {Config.DATA.TASK_NAME}")
    
    # Example for the MNLI task (Accuracy metric)
    if Config.DATA.TASK_NAME == "mnli":
        # Simulate predictions and labels (MNLI is a 3-class task: 0, 1, 2)
        sim_predictions = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 1])
        sim_labels =      np.array([0, 0, 2, 0, 1, 1, 2, 0, 1, 1])
        
        # 7 out of 10 correct: (0, 2, 0, 1, 1, 0, 1) are matches
        # Correct indices: 0, 2, 3, 4, 8, 9 
        # Correct count: 6
        # Accuracy expected: 6/10 = 0.6
        
        sim_eval_preds = EvalPrediction(predictions=np.zeros_like(sim_predictions), label_ids=sim_labels)
        
        # To make the test runnable, we'll quickly compute the argmax simulation
        # In a real test, the predictions argument is the LOGIT array.
        
        # Create a mock logits array where the argmax gives sim_predictions
        mock_logits = np.zeros((10, 3))
        mock_logits[np.arange(10), sim_predictions] = 1.0 # Set the predicted class logit high

        results = compute_metrics(EvalPrediction(predictions=mock_logits, label_ids=sim_labels))
        
        print(f"\nComputed Metrics (Simulated): {results}")