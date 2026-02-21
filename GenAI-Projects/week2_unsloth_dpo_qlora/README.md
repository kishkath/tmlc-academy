# README.md

## Fitness QA Fine-Tuning Project

---

### Main Motto of the Project

The goal of this project is to create a **specialized fitness question-answering bot** that can understand nuanced questions and provide accurate, context-aware answers. This is achieved by fine-tuning a large language model specifically for fitness-related queries using efficient methods such as LoRA and DPO.

---

### Detailed Processing Workflow

**Workflow:** Dataset Preparation → Model Loading (Unsloth Qwen3) → LoRA Fine-Tuning with DPO → wandb Logging → Save Fine-Tuned Model → Inference

| Step                 | Description                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------- |
| Dataset Preparation  | Load, clean, and preprocess question-answer data for controlled and quality training.        |
| Model Selection      | Use Unsloht’s Qwen3-0.6B for efficiency, quantization support, and scalability.              |
| Fine-Tuning Strategy | Apply DPO for preference optimization and LoRA for parameter-efficient tuning.               |
| Training Process     | Configure batch size, learning rate, gradient accumulation, and logging for stable training. |
| Inference            | Load fine-tuned model with flexible config and generate context-aware answers.               |

---

### Trainer Configuration

| Resource                    | Configured Value | Description                                                                                                                                                                   |
| --------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| per_device_train_batch_size | 2                | Keeps memory low, enabling training on limited GPU resources.                                                                                                                 |
| gradient_accumulation_steps | 4                | Simulates larger batch sizes for stability without increasing memory usage.                                                                                                   |
| warmup_ratio                | 0.1              | Gradual ramp-up of learning rate for improved convergence.                                                                                                                    |
| num_train_epochs            | 4                | Balanced epochs to avoid overfitting for a smaller dataset.                                                                                                                   |
| learning_rate               | 5e-6             | Low learning rate to maintain pretrained knowledge during fine-tuning.                                                                                                        |
| logging_steps               | 10               | Frequent logging to monitor training without excessive overhead.                                                                                                              |
| optim                       | adamw_8bit       | Memory-efficient optimizer, ideal for quantized models.                                                                                                                       |
| weight_decay                | 0.01             | Regularization to prevent overfitting.                                                                                                                                        |
| lr_scheduler_type           | linear           | Steady learning rate decay for smooth training.                                                                                                                               |
| seed                        | 42               | Ensures reproducibility of results.                                                                                                                                           |
| output_dir                  | outputs          | Directory to save trained models and tokenizer.                                                                                                                               |
| beta                        | 0.1              | DPO parameter controlling preference optimization strength.                                                                                                                   |
| max_length                  | 1024             | Maximum sequence length for generation, ensuring sufficient space for prompts and responses without exhausting resources.                                                     |
| max_prompt_length           | 256              | Limits prompt token length to optimize memory usage while preserving context. This is important for balancing training efficiency and maintaining relevant input information. |

---

### Prompt and Length Considerations

* **Prompt Size**: Keeping prompts within 256 tokens helps maintain context relevance while avoiding memory overflow.
* **max_length**: Set to 1024 tokens to accommodate longer queries and answers without trimming crucial context, but also keeping resource demands manageable.
* **Adaptive Generation**: Dynamically adjusts generation length based on available VRAM, ensuring inference remains stable across CPU and GPU environments.

---

### Why These Choices

* **Unsloht Models**: Optimized for performance and memory efficiency.
* **DPO**: Aligns models with preference-based tasks.
* **LoRA**: Enables parameter-efficient fine-tuning.
* **4-bit Quantization**: Allows running large models in resource-constrained environments.
* **wandb Integration**: Ensures comprehensive tracking and reproducibility.

---
### DPO Training Metrics Overview

| Parameter              | Description                                                                                                                                                             | Importance                                                                                 | Desired Status                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **loss**               | The Direct Preference Optimization loss. It measures how well the model aligns with preference data (chosen vs rejected). Lower loss means the model is better aligned. | Indicates convergence — a key sign that training is succeeding.                            | **Lower is better**                                                                          |
| **grad_norm**          | Norm of gradients during backpropagation. Shows the magnitude of updates applied to model weights.                                                                      | Helps detect instability or exploding/vanishing gradients.                                 | **Lower but not zero** (small and stable)                                                    |
| **learning_rate**      | Step size for each optimization update. Controls how quickly the model learns.                                                                                          | Balances learning speed and stability. Too high → instability; too low → slow convergence. | **Depends on stage** (decays over time)                                                      |
| **rewards/chosen**     | Average reward for the chosen responses from the reward model.                                                                                                          | Shows how much the model prefers chosen outputs.                                           | **Higher is better**                                                                         |
| **rewards/rejected**   | Average reward for the rejected responses from the reward model.                                                                                                        | Used to measure how well the model penalizes undesirable responses.                        | **Lower is better**                                                                          |
| **rewards/accuracies** | Fraction of preference comparisons where the chosen response was ranked better than the rejected one.                                                                   | Direct indicator of preference alignment.                                                  | **Higher is better (max = 1.0)**                                                             |
| **rewards/margins**    | Difference between chosen and rejected rewards.                                                                                                                         | Measures separation between preferred and non-preferred responses.                         | **Higher is better**                                                                         |
| **logps/chosen**       | Sum of log-probabilities assigned to the chosen sequence.                                                                                                               | Helps measure how confidently the model generates chosen responses.                        | **Higher per token is better** (note: absolute values must be normalized by sequence length) |
| **logps/rejected**     | Sum of log-probabilities assigned to the rejected sequence.                                                                                                             | Used to compare against chosen log-probabilities to calculate preference.                  | **Lower per token is better**                                                                |
| **logits/chosen**      | Raw output scores before normalization for chosen sequences.                                                                                                            | Shows model’s unnormalized confidence for chosen outputs.                                  | **Higher relative to rejected**                                                              |
| **logits/rejected**    | Raw output scores before normalization for rejected sequences.                                                                                                          | Used for comparison in preference modeling.                                                | **Lower relative to chosen**                                                                 |
| **epoch**              | Current training iteration over the dataset.                                                                                                                            | Tracks training progress and convergence.                                                  | **Increases during training**                                                                |


This configuration ensures a robust, scalable, and efficient fine-tuning pipeline suitable for domain-specific QA bot development while maintaining interpretability and resource efficiency.
