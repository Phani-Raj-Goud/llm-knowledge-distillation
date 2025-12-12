# Knowledge Distillation of LLMs on the GLUE Benchmark

## Project Aim

This project implements **Knowledge Distillation (KD)** to create a smaller, faster, and more efficient language model (the **Student**) that retains the high performance of a much larger, state-of-the-art model (the **Teacher**).

We utilize the **GLUE (General Language Understanding Evaluation)** benchmark for a sequence classification task (default is **MNLI**) to demonstrate the successful transfer of task-specific knowledge.

### **The Goal**

To distill the task expertise from a large, resource-intensive model (e.g., `bert-large-uncased`) into a compact, deployment-friendly model (e.g., `bert-base-uncased`) with minimal loss in accuracy. 

## Project Structure

The core logic is modularized in the `src/` directory:

| Component | Role |
| :--- | :--- |
| `src/distillation_loss.py` | Defines the custom loss function ($L = \alpha \cdot L_{KL} + (1-\alpha) \cdot L_{CE}$). |
| `src/models.py` | Handles loading and freezing the Teacher model and initializing the Student model. |
| `src/trainer.py` | Contains the custom `DistillationTrainer` subclass for executing the KD training loop. |
| `main.py` | Orchestrates the entire pipeline, from data preparation (including Teacher Logit generation) to final model saving. |
| `inference.py` | Script to use the final distilled model for real-time predictions. |

## Getting Started

### 1. Prerequisites

Ensure you have Python (3.8+) installed.

### 2. Setup Environment

Clone the repository (or create the folder structure) and install dependencies:

```bash
git clone https://github.com/Phani-Raj-Goud/llm-knowledge-distillation.git
cd knowledge_distillation_project
pip install -r requirements.txt