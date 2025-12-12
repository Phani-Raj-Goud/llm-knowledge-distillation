import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from config import Config # Import our centralized configuration

def load_teacher_model():
    """
    Loads the pre-trained Teacher model and configures it to output logits
    required for knowledge distillation.

    The teacher model is usually frozen during distillation, and its primary role
    is to provide the soft targets (logits).
    """
    teacher_name = Config.MODEL.TEACHER_MODEL_NAME
    num_labels = Config.DATA.NUM_LABELS
    
    print(f"Loading Teacher Model: {teacher_name}")
    
    # Load the configuration first
    config = AutoConfig.from_pretrained(
        teacher_name,
        num_labels=num_labels
    )
    
    # Load the pre-trained model for sequence classification
    # We explicitly tell it the number of labels for the GLUE task
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher_name,
        config=config
    )
    
    # --- IMPORTANT DISTILLATION PREPARATION ---
    # 1. Freeze the teacher model's parameters (optional but recommended)
    #    The teacher is an expert and should not be modified during student training.
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Set model to evaluation mode
    model.eval()
    
    print(f"Teacher Model loaded successfully. Total parameters: {model.num_parameters()}")
    
    return model

def load_student_model():
    """
    Loads the pre-trained Student model and configures it for fine-tuning
    via knowledge distillation.

    The student model's parameters will be updated during the distillation process.
    """
    student_name = Config.MODEL.STUDENT_MODEL_NAME
    num_labels = Config.DATA.NUM_LABELS
    
    print(f"Loading Student Model: {student_name}")
    
    # Load the configuration for the student
    config = AutoConfig.from_pretrained(
        student_name,
        num_labels=num_labels,
        # Set dropout higher for distillation, as it helps prevent overfitting
        # when relying heavily on the soft targets.
        hidden_dropout_prob=Config.MODEL.CLASSIFIER_DROPOUT,
        attention_probs_dropout_prob=Config.MODEL.CLASSIFIER_DROPOUT
    )

    # Load the pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        student_name,
        config=config
    )
    
    # Ensure the model is in training mode initially
    model.train()
    
    print(f"Student Model loaded successfully. Total parameters: {model.num_parameters()}")
    
    return model

# Optional test block to ensure models load correctly
if __name__ == '__main__':
    # Set device if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and check Teacher Model
    teacher = load_teacher_model().to(device)
    print(f"Teacher device: {next(teacher.parameters()).device}")
    
    # Load and check Student Model
    student = load_student_model().to(device)
    print(f"Student device: {next(student.parameters()).device}")

    # Check that teacher parameters are frozen (should print False)
    teacher_frozen = all(not p.requires_grad for p in teacher.parameters())
    print(f"Teacher parameters frozen: {teacher_frozen}")