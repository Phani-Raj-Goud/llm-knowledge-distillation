from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

def load_and_process_glue_data():
    # 1. Load the raw dataset
    raw_datasets = load_dataset(config.DATASET_NAME, config.TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_NAME) # Use student tokenizer

    # 2. Tokenization function
    def tokenize_function(examples):
        # GLUE tasks have different column names (e.g., 'sentence1', 'sentence2')
        # We need a function to handle the specific task structure.
        if config.TASK_NAME == "mnli":
            text_a, text_b = examples["premise"], examples["hypothesis"]
        else: # For single sentence tasks like SST-2
            text_a, text_b = examples["sentence"], None
            
        return tokenizer(text_a, text_b, truncation=True, padding="max_length")

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    
    # 3. Add Teacher Logits (THIS IS THE CRITICAL STEP)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(config.TEACHER_MODEL_NAME)
    
    def generate_teacher_logits(examples):
        # Convert examples to PyTorch/TensorFlow and get logits
        # (Simplified for illustration; needs proper batching in practice)
        inputs = {k: examples[k] for k in ["input_ids", "attention_mask"]}
        
        # Disable gradient calculation for teacher model
        with torch.no_grad():
             outputs = teacher_model(**inputs)
        
        # Save the logits as a new column
        examples["teacher_logits"] = outputs.logits.cpu().numpy()
        return examples

    # Apply the logit generation to the training split
    train_dataset_with_logits = tokenized_datasets["train"].map(generate_teacher_logits, batched=True, batch_size=32)
    
    # Select and rename columns for the trainer
    train_dataset_with_logits = train_dataset_with_logits.rename_column("label", "hard_labels")
    train_dataset_with_logits = train_dataset_with_logits.remove_columns(["premise", "hypothesis", "idx"])
    
    return train_dataset_with_logits, tokenized_datasets["validation"]