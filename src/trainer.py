import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from config import Config
from .models import load_teacher_model, load_student_model
from .distillation_loss import DistillationLoss
from typing import Dict, Union, Any, Optional, List, Tuple
from datasets import Dataset

# --- 1. Define the Custom Distillation Trainer Class ---

class DistillationTrainer(Trainer):
    """
    A custom Hugging Face Trainer subclass that implements the Knowledge 
    Distillation loss function.

    This class overrides the default 'compute_loss' method to calculate 
    the combined Hard (CE) and Soft (KL) loss.
    """
    def __init__(
        self,
        teacher_model: torch.nn.Module = None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Store the teacher model and send it to the correct device
        self.teacher = teacher_model.to(self.args.device)
        # Ensure teacher is in evaluation mode and frozen
        self.teacher.eval() 

        # Initialize the custom distillation loss criterion
        self.distillation_criterion = DistillationLoss()
        
    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Custom method to compute the knowledge distillation loss.
        """
        # The teacher logits are expected to be in the dataset inputs
        if "teacher_logits" not in inputs:
            raise KeyError("The training batch must contain a 'teacher_logits' column for distillation.")

        # Extract teacher logits and ground truth labels
        teacher_logits = inputs.pop("teacher_logits")
        labels = inputs.pop("labels")

        # Forward pass for the student model
        # The student model is passed to the parent Trainer's compute_loss 
        # as 'model'.
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # --- Calculate the Distillation Loss ---
        
        # Check that the number of classes matches
        if student_logits.shape != teacher_logits.shape:
             raise ValueError("Student and Teacher logits must have the same shape.")

        # Calculate the total loss and individual components
        total_loss, loss_parts = self.distillation_criterion(
            student_logits=student_logits, 
            labels=labels, 
            teacher_logits=teacher_logits
        )
        
        # Log the components of the loss for monitoring (using an internal callback)
        # Note: This requires a custom callback or manually logging the components.
        # For simplicity here, we rely on the Trainer's primary loss reporting.

        if return_outputs:
            # We return the total loss and the outputs (which include student logits)
            return total_loss, student_outputs
        else:
            return total_loss

# --- 2. Main Training Execution Function (for main.py to call) ---

def run_distillation_training(
    student_model: torch.nn.Module, 
    teacher_model: torch.nn.Module, 
    train_dataset: Dataset, 
    eval_dataset: Dataset,
    compute_metrics: Optional[Any] = None
):
    """
    Sets up and executes the Knowledge Distillation training process.
    """
    # 1. Define Training Arguments using Config
    training_args = TrainingArguments(
        output_dir=Config.PATHS.STUDENT_MODEL_SAVE,
        overwrite_output_dir=True,
        num_train_epochs=Config.TRAINING.NUM_EPOCHS,
        per_device_train_batch_size=Config.TRAINING.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.TRAINING.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=Config.TRAINING.LEARNING_RATE,
        weight_decay=Config.TRAINING.WEIGHT_DECAY,
        logging_dir=Config.PATHS.RESULTS_DIR,
        logging_steps=Config.TRAINING.LOGGING_STEPS,
        evaluation_strategy=Config.TRAINING.EVALUATION_STRATEGY,
        save_strategy=Config.TRAINING.SAVE_STRATEGY,
        load_best_model_at_end=Config.TRAINING.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="accuracy" # Assuming accuracy is the key metric
    )

    # 2. Initialize the Distillation Trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 3. Start Training
    print("Starting Distillation Training...")
    train_result = trainer.train()

    # 4. Save the Final Model and Metrics
    trainer.save_model(Config.PATHS.STUDENT_MODEL_SAVE)
    
    # 5. Evaluate the best model
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    return train_result, metrics
