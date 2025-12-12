import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class DistillationLoss(nn.Module):
    """
    The combined Knowledge Distillation Loss function.

    It linearly combines the standard Cross-Entropy Loss (Hard Targets) 
    and the temperature-scaled Kullback-Leibler Divergence (Soft Targets).
    """
    def __init__(self):
        super().__init__()
        
        # Load hyperparameters from the global config
        self.temperature = Config.DISTILLATION.TEMPERATURE
        self.alpha = Config.DISTILLATION.ALPHA
        
        # Criterion for the Hard Target Loss (Cross-Entropy with true labels)
        self.hard_criterion = nn.CrossEntropyLoss()
        
        # Criterion for the Soft Target Loss (Kullback-Leibler Divergence)
        # We use reduction='batchmean' as it is the mathematically correct
        # way to compute the mean KL divergence over a batch.
        # It is crucial to set log_target=False (default) or ensure the 
        # target (Teacher's soft probability) is NOT in log space.
        self.kl_div_criterion = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, labels, teacher_logits):
        """
        Calculates the total distillation loss.

        Args:
            student_logits (torch.Tensor): Logits output by the Student model.
            labels (torch.Tensor): Ground-truth integer labels (Hard Targets).
            teacher_logits (torch.Tensor): Logits output by the Teacher model 
                                          (Soft Targets).

        Returns:
            torch.Tensor: The total calculated distillation loss.
            dict: Dictionary containing the individual loss components.
        """
        T = self.temperature
        
        # 1. Hard Target Loss (L_CE)
        # Standard Cross-Entropy on the student's raw logits and hard labels
        # This term ensures the student is still learning the correct task.
        loss_hard = self.hard_criterion(student_logits, labels)
        
        # 2. Soft Target Loss (L_KL)
        
        # Step 2a: Calculate Teacher's Soft Probabilities (P_t)
        # Use softmax with temperature T. The teacher's probabilities serve as the TARGET.
        # We detach the tensor to ensure no gradients flow back into the Teacher model.
        teacher_soft_probs = F.softmax(teacher_logits / T, dim=-1).detach()
        
        # Step 2b: Calculate Student's Soft Log-Probabilities (log(P_s))
        # KLDivLoss expects the INPUT (Student) to be in log-probability space.
        student_soft_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        # Step 2c: Calculate KL Divergence
        # KL(P_t || P_s) = P_t * log(P_t / P_s) = P_t * (log(P_t) - log(P_s))
        loss_soft = self.kl_div_criterion(student_soft_log_probs, teacher_soft_probs)
        
        # Step 2d: Apply the T^2 scaling factor
        loss_soft = loss_soft * (T * T)
        
        # 3. Combined Loss
        # L = (1-\alpha) * L_CE + \alpha * L_KL
        total_loss = (1.0 - self.alpha) * loss_hard + self.alpha * loss_soft
        
        # Return total loss and individual components for logging
        return total_loss, {
            'total_loss': total_loss.item(),
            'hard_loss': loss_hard.item(),
            'soft_loss': loss_soft.item()
        }

# Test block
if __name__ == '__main__':
    print("Testing DistillationLoss class with dummy data...")
    
    # Create dummy logits (e.g., batch_size=4, num_labels=3)
    dummy_student_logits = torch.randn(4, 3, requires_grad=True)
    dummy_teacher_logits = torch.randn(4, 3) + 5.0 # Teacher is more confident/higher logit range
    dummy_labels = torch.randint(0, 3, (4,))

    # Initialize the loss function
    kd_criterion = DistillationLoss()
    
    # Calculate loss
    total_loss, loss_parts = kd_criterion(
        dummy_student_logits, 
        dummy_labels, 
        dummy_teacher_logits
    )
    
    # Print results
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Hard Loss (CE): {loss_parts['hard_loss']:.4f}")
    print(f"Soft Loss (KL): {loss_parts['soft_loss']:.4f}")
    
    # Verify gradient can be calculated
    total_loss.backward()
    print("Successfully calculated gradients (loss.backward()).")