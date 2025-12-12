import os
import argparse
import torch
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset, Dataset
import numpy as np
import json
from config import Config
from src.models import load_teacher_model, load_student_model
from src.trainer import run_distillation_training
from src.evaluation import compute_metrics # We use the function defined in evaluation.py

# --- 1. Environment Setup and Data Preparation ---

def setup_environment():
    """Creates necessary directories and sets the device."""
    
    # Create output directories if they don't exist
    os.makedirs(Config.PATHS.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.PATHS.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.PATHS.DATA_DIR, exist_ok=True)

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return device

def load_and_prepare_data(device: torch.device):
    """
    Loads, tokenizes, and adds teacher logits to the GLUE dataset.
    This combines the steps previously discussed for data_loader.py for simplicity.
    """
    # 1. Load raw data and tokenizer
    print(f"Loading GLUE dataset: {Config.DATA.TASK_NAME}")
    raw_datasets = load_dataset(Config.DATA.DATASET_NAME, Config.DATA.TASK_NAME)
    
    # Use the Student's tokenizer as the student is the model being fine-tuned
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL.STUDENT_MODEL_NAME) 

    # 2. Tokenization function (must handle different GLUE column names)
    def tokenize_function(examples):
        # Logic to handle single-sentence (SST-2) vs. sentence-pair (MNLI) tasks
        sentence1_key = "sentence" if "sentence" in examples else "sentence1"
        sentence2_key = None if "sentence" in examples else "sentence2"
        
        # MNLI uses 'premise' and 'hypothesis' instead of 'sentence1'/'sentence2'
        if Config.DATA.TASK_NAME == "mnli":
            sentence1_key = "premise"
            sentence2_key = "hypothesis"

        text_a = examples[sentence1_key]
        text_b = examples[sentence2_key] if sentence2_key else None
        
        return tokenizer(
            text_a, 
            text_b, 
            truncation=True, 
            padding="max_length", 
            max_length=Config.DATA.MAX_SEQ_LENGTH
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names # Remove raw text columns
    )
    
    # Rename 'label' column to 'labels' for Hugging Face Trainer compatibility
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 3. Generate Teacher Logits (The critical KD step)
    print(f"Generating Teacher Logits using {Config.MODEL.TEACHER_MODEL_NAME}...")
    teacher_model = load_teacher_model().to(device)

    def generate_teacher_logits(examples):
        # Convert examples into a format the model can use (PyTorch tensors)
        inputs = {
            "input_ids": torch.tensor(examples["input_ids"], dtype=torch.long).to(device),
            "attention_mask": torch.tensor(examples["attention_mask"], dtype=torch.long).to(device)
        }
        
        # Run inference (no gradient calculation)
        with torch.no_grad():
             outputs = teacher_model(**inputs)
        
        # Save the raw logits as a new numpy array column
        examples["teacher_logits"] = outputs.logits.cpu().numpy()
        return examples

    # Apply logit generation to the training split
    train_dataset = tokenized_datasets["train"].map(
        generate_teacher_logits, 
        batched=True, 
        batch_size=Config.TRAINING.PER_DEVICE_EVAL_BATCH_SIZE, # Use large batch size for inference speed
        desc="Generating Teacher Logits"
    )
    
    # Select the correct validation set for MNLI
    eval_key = "validation_matched" if Config.DATA.TASK_NAME == "mnli" else "validation"
    eval_dataset = tokenized_datasets[eval_key]

    # Clean up teacher model/memory after use
    del teacher_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Data preparation complete. Teacher logits added to training set.")
    return train_dataset, eval_dataset

# --- 2. Main Execution Block ---

def main(args):
    """Main function to run the Knowledge Distillation pipeline."""
    
    device = setup_environment()
    
    # 1. Load and Prepare Data (including Teacher Logits generation)
    train_dataset, eval_dataset = load_and_prepare_data(device)

    # 2. Load Models
    teacher_model = load_teacher_model().to(device)
    student_model = load_student_model()
    
    # 3. Run Distillation Training
    print("\n--- Starting Knowledge Distillation Training ---\n")
    

    # The teacher model is passed to the trainer for use in the compute_loss method
    train_result, eval_metrics = run_distillation_training(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    print("\n--- Distillation Training Complete ---\n")
    
    # 4. Final Cleanup and Reporting
    print(f"Best model saved to: {Config.PATHS.STUDENT_MODEL_SAVE}")
    print(f"Final Evaluation Metrics:")
    print(json.dumps(eval_metrics, indent=4))
    
    # Save final metrics to file
    with open(Config.PATHS.EVAL_RESULTS_FILE, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    print(f"Metrics saved to {Config.PATHS.EVAL_RESULTS_FILE}")


if __name__ == "__main__":
    # You can add argument parsing here to easily override config parameters
    # For simplicity, we are running directly with default config values.
    
    # Example Argument Parser (Optional but Recommended)
    parser = argparse.ArgumentParser(description="Knowledge Distillation of LLMs on GLUE.")
    parser.add_argument("--alpha", type=float, default=Config.DISTILLATION.ALPHA, help="Weight for KL divergence loss.")
    parser.add_argument("--temperature", type=float, default=Config.DISTILLATION.TEMPERATURE, help="Temperature for softening logits.")
    # In a full project, you would update the Config class based on parser arguments.
    
    args = parser.parse_args()
    
    # Quick check for MNLI validation split consistency
    if Config.DATA.TASK_NAME == "mnli":
        print("Note: Using the 'validation_matched' split for MNLI evaluation.")
        
    main(args)