import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
from typing import List, Dict

# --- 1. Load Mappings for Human-Readable Output ---

def get_label_mapping(task_name: str) -> Dict[int, str]:
    """
    Returns the integer-to-string label mapping for a given GLUE task.
    This is necessary to present human-readable results.
    """
    if task_name == "mnli":
        # Standard mapping for MNLI: 0=entailment, 1=neutral, 2=contradiction
        return {0: "entailment", 1: "neutral", 2: "contradiction"}
    elif task_name == "sst2":
        # SST-2 (Sentiment): 0=negative, 1=positive
        return {0: "negative", 1: "positive"}
    # Add other GLUE tasks as needed (e.g., CoLA, QQP, MRPC)
    else:
        # Default for binary classification if not explicitly defined
        return {0: "class_0", 1: "class_1"} 

# --- 2. The Core Inference Function ---

class DistilledModelInference:
    """
    A class to handle loading and inference for the distilled Student model.
    """
    def __init__(self):
        # Determine the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading inference model on device: {self.device}")
        
        # Load the task-specific label mapping
        self.label_map = get_label_mapping(Config.DATA.TASK_NAME)
        
        # Load the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.PATHS.STUDENT_MODEL_SAVE)
        
        # Load the Distilled Student Model
        # It must be loaded from the saved checkpoint path
        self.model = AutoModelForSequenceClassification.from_pretrained(Config.PATHS.STUDENT_MODEL_SAVE)
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        print(f"Distilled Student Model loaded from: {Config.PATHS.STUDENT_MODEL_SAVE}")

    def predict_single(self, text_a: str, text_b: str = None) -> Dict[str, str]:
        """
        Runs inference for a single text input or text pair.
        """
        # 1. Tokenize input
        inputs = self.tokenizer(
            text_a, 
            text_b, 
            padding=True, 
            truncation=True, 
            max_length=Config.DATA.MAX_SEQ_LENGTH, 
            return_tensors="pt"
        )
        
        # Move tensors to the appropriate device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Run Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 3. Process Output Logits
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().squeeze()
        
        # Get the predicted class index
        predicted_index = torch.argmax(probabilities).item()
        
        # 4. Format Result
        result = {
            "prediction": self.label_map.get(predicted_index, "Unknown Class"),
            "probability": f"{probabilities[predicted_index].item():.4f}",
            "raw_probabilities": {self.label_map[i]: f"{p.item():.4f}" for i, p in enumerate(probabilities)}
        }
        
        return result

# --- 3. Command Line Interface (CLI) Execution ---

def main():
    parser = argparse.ArgumentParser(description=f"Run inference on the distilled Student Model ({Config.MODEL.STUDENT_MODEL_NAME}).")
    parser.add_argument("--text_a", type=str, required=True, help="The primary input text (e.g., sentence for SST-2 or premise for MNLI).")
    parser.add_argument("--text_b", type=str, default=None, help="The secondary input text (e.g., hypothesis for MNLI). Required for sentence-pair tasks.")
    
    args = parser.parse_args()

    # Check if the required second sentence is provided for pair tasks (like MNLI)
    is_pair_task = Config.DATA.TASK_NAME in ["mnli", "qqp", "mrpc"]
    if is_pair_task and args.text_b is None:
        print(f"Error: Task '{Config.DATA.TASK_NAME}' requires two text inputs (text_a and text_b).")
        return

    # Initialize the inference class
    try:
        inference_engine = DistilledModelInference()
    except Exception as e:
        print(f"Failed to load the model. Ensure training completed successfully and the path is correct: {Config.PATHS.STUDENT_MODEL_SAVE}")
        print(f"Error: {e}")
        return

    # Run prediction
    print("\n--- Running Inference ---")
    prediction_result = inference_engine.predict_single(args.text_a, args.text_b)
    
    # Print results
    print(f"\nTask: {Config.DATA.TASK_NAME.upper()}")
    print("-" * 30)
    print(f"Input A: {args.text_a}")
    if args.text_b:
        print(f"Input B: {args.text_b}")
    print("-" * 30)
    print(f"Predicted Class: **{prediction_result['prediction']}**")
    print(f"Confidence: {prediction_result['probability']}")
    print("Full Probability Distribution:")
    for label, prob in prediction_result['raw_probabilities'].items():
        print(f"  - {label}: {prob}")

if __name__ == "__main__":
    main()