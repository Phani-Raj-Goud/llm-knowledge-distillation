

class PathConfig:
    """Configuration for all file paths and directories."""
    ROOT_DIR = "."
    DATA_DIR = "data"
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    
    # Processed Data Files
    TRAIN_FILE = f"{DATA_DIR}/train_with_logits.pt"
    VALID_FILE = f"{DATA_DIR}/validation_processed.pt"
    
    # Model Save Paths
    TEACHER_MODEL_SAVE = f"{MODELS_DIR}/teacher_model"
    STUDENT_MODEL_SAVE = f"{MODELS_DIR}/student_model_distilled"
    
    # Results Paths
    LOG_FILE = f"{RESULTS_DIR}/training_log.csv"
    EVAL_RESULTS_FILE = f"{RESULTS_DIR}/evaluation_metrics.json"


class DataConfig:
    """Configuration for the dataset and preprocessing."""
    
    # The Hugging Face dataset name
    DATASET_NAME = "glue"
    
    # The specific task to use from the benchmark (e.g., MNLI, SST2, QQP)
    # MNLI (Multi-Genre Natural Language Inference) is a popular, 3-class task
    TASK_NAME = "mnli"
    
    # Maximum sequence length for tokenization
    MAX_SEQ_LENGTH = 128
    
    # Number of classes in the chosen task (MNLI has 3: entailment, neutral, contradiction)
    # This must match the model's num_labels when loading for sequence classification
    NUM_LABELS = 3


class ModelConfig:
    """Configuration for Teacher and Student model architectures."""
    
    # Teacher Model: A large, high-performing model (e.g., BERT-Large)
    TEACHER_MODEL_NAME = "bert-large-uncased"
    
    # Student Model: A smaller, more efficient model (e.g., BERT-Base or DistilBERT)
    # We use BERT-base here to ensure full feature compatibility with BERT-Large
    STUDENT_MODEL_NAME = "bert-base-uncased" 
    
    # Dropout probability for the student classifier head (often increased for KD)
    CLASSIFIER_DROPOUT = 0.2 


class DistillationConfig:
    """Hyperparameters specific to the Knowledge Distillation process."""
    
    # Temperature (T or Tau): Controls the 'softness' of the teacher's logits.
    # Recommended range is often 2.0 to 5.0. T=2.0 is a strong starting point.
    TEMPERATURE = 2.0
    
    # Alpha (α): Weight for the KL Divergence Loss (Soft Targets).
    # Total Loss = (α * KL_Loss) + ((1-α) * CrossEntropy_Loss)
    # A common starting point is to weigh both losses equally (0.5), 
    # but some literature suggests favoring KL (e.g., 0.9 or higher).
    ALPHA = 0.5 


class TrainingConfig:
    """Standard training loop hyperparameters."""
    
    # Optimization
    LEARNING_RATE = 5e-5       # Standard fine-tuning learning rate for BERT
    WEIGHT_DECAY = 0.01        # L2 regularization
    ADAM_EPSILON = 1e-6        # Epsilon for the Adam optimizer
    
    # Training Loop
    NUM_EPOCHS = 5             # Distillation often converges faster than full fine-tuning
    PER_DEVICE_TRAIN_BATCH_SIZE = 32
    PER_DEVICE_EVAL_BATCH_SIZE = 64
    GRADIENT_ACCUMULATION_STEPS = 1 # Use 1 unless memory is a major constraint
    
    # Logging and Checkpointing
    LOGGING_STEPS = 100
    EVALUATION_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LOAD_BEST_MODEL_AT_END = True

class Config:
    """Main container for all project configuration settings."""
    PATHS = PathConfig
    DATA = DataConfig
    MODEL = ModelConfig
    DISTILLATION = DistillationConfig
    TRAINING = TrainingConfig

# Simple command to test or print config settings (optional)
if __name__ == '__main__':
    print(f"--- Project Configuration Loaded ---")
    print(f"Teacher Model: {Config.MODEL.TEACHER_MODEL_NAME}")
    print(f"Student Model: {Config.MODEL.STUDENT_MODEL_NAME}")
    print(f"Distillation Alpha (alpha): {Config.DISTILLATION.ALPHA}")
    print(f"Distillation Temperature (T): {Config.DISTILLATION.TEMPERATURE}")