# Qwen LoRA Fine-Tuning for Fitness QA Bot

## Project Overview

This project fine-tunes the **Qwen2.5-0.5B Instruct model** on a domain-specific **fitness & nutrition FAQ dataset** using **LoRA (Low-Rank Adaptation)**. LoRA allows parameter-efficient fine-tuning, meaning we only train a small number of extra parameters while keeping the large base model frozen. This is memory-efficient, faster, and works well with small datasets.

* **Training data:** 5,000 examples
* **Validation data:** 800 examples
* **Goal:** Provide accurate answers for fitness & nutrition questions

---

## Directory Structure

```
root/
├── train.py                 # Fine-tuning script
├── inference.py             # Inference script
├── configurations/
│   └── config.json          # Single configuration file for training & inference
├── dataset/
│   ├── qa_fitness_nutrition_train.jsonl
│   └── qa_fitness_nutrition_val.jsonl
├── finetuning/
│   ├── model.py
│   └── trainer.py
├── dataset/
│   └── data_utils.py
├── requirements.txt
└── README.md
```

---

## Configuration (`config.json`)

We use a **single JSON file** to hold all configurations for training and inference. This allows easy tuning without changing Python code.

### Device

```json
"device": "cuda"
```

* Uses GPU if available for efficient training.

### Model

```json
"model_name": "Qwen/Qwen2.5-0.5B-Instruct"
```

* Instruction-tuned base model for better QA performance.
* Only the LoRA adapter is fine-tuned.

### Dataset

```json
"train_file": "dataset/qa_fitness_nutrition_train.jsonl",
"eval_file": "dataset/qa_fitness_nutrition_val.jsonl",
"max_length": 512
```

* Training: 5,000 examples, Validation: 800 examples.
* `max_length=512` balances context coverage vs memory usage.

### LoRA Configuration

```json
"lora": {
  "r": 16,
  "lora_alpha": 64,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.15,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

* **Rank (`r=16`)**: Low-rank factor; small, memory-efficient.
* **Alpha (`lora_alpha=64`)**: Scaling for LoRA weights.
* **Target modules**: `q_proj`, `v_proj` are most impactful for causal LM.
* **Dropout (`0.15`)**: Prevents overfitting.
* **Bias=`none`**: Simplifies model, reduces trainable parameters.

### Training Parameters

```json
"training": {
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 16,
  "learning_rate": 3e-5,
  "num_train_epochs": 5,
  "warmup_steps": 50,
  "logging_steps": 20,
  "evaluation_strategy": "steps",
  "eval_steps": 50,
  "save_strategy": "steps",
  "save_steps": 50,
  "save_total_limit": 2,
  "output_dir": "./qwen2.5-0.5b-finetuned",
  "logging_dir": "./logs",
  "load_best_model_at_end": true,
  "metric_for_best_model": "loss",
  "report_to": "none"
}
```

* **Effective batch size = 32** (2 x 16 gradient accumulation) for stable training.
* **Learning rate = 3e-5** for stable convergence on small dataset.
* **Evaluation every 50 steps** monitors validation loss.
* **Load best model at end** automatically selects the checkpoint with lowest validation loss.

### Inference Parameters

```json
"inference": {
  "mode": "lora",
  "base_model_name": "Qwen/Qwen2.5-0.5B",
  "adapter_path": "./qwen2.5-0.5b-finetuned",
  "merged_model_path": "./qwen2.5-0.5b-merged",
  "generation": {
    "max_new_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.9
  },
  "questions": [
    "What are the benefits of regular exercise?",
    "How often should I do strength training?",
    "What is a healthy diet plan for weight loss?"
  ]
}
```

* **LoRA mode** → inference uses only fine-tuned adapter.
* **Merged mode optional** → combines LoRA weights with base model for faster inference.
* **Temperature = 0.2** → deterministic, factual answers.
* **Top-p = 0.9** → restricts outputs to likely tokens.

---

## Training Instructions

```bash
python train.py
```

* Loads model, tokenizer, and dataset.
* Tokenizes data in format:

```
User: {question}
Assistant: {answer}
```

* Initializes **SFTTrainer** with LoRA config.
* Trains model with gradient accumulation, evaluates periodically, and saves LoRA adapter weights.

---

## Inference Instructions

```bash
python inference.py
```

* Supports **LoRA-only** or **merged model** inference.
* Uses `questions` list from `config.json`.
* Logs both questions and generated answers.

**Example Output:**

```
Question: What are the benefits of regular exercise?
Answer: Regular exercise improves cardiovascular health, increases strength and flexibility, boosts mood, and aids in weight management.
```

---

## Advantages of This Setup

1. Parameter-efficient LoRA fine-tuning → minimal GPU memory usage.
2. Small dataset friendly → only 5,000 examples required.
3. Early stopping & best checkpoint selection → prevents overfitting.
4. Structured prompts → consistent output format.
5. Flexible inference → adapter-only or merged model.
6. Single config file → easy maintenance and experimentation.

---

## Deployment Recommendations

* Free GPU instances (Kaggle, Colab) can handle training and inference.
* `max_length=512` ensures efficient inference.
* Use LoRA adapter for small deployment footprint (\~100-200MB).
* Merged model recommended for production inference if GPU memory allows.

---

## Requirements (`requirements.txt`)

```
torch>=2.2.0
transformers>=4.34.0
datasets>=2.15.1
peft>=0.6.0
trl>=0.22.2
accelerate>=0.22.0
```

Install via:

```bash
pip install -r requirements.txt
```
