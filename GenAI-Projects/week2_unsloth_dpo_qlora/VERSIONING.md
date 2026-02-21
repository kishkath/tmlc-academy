| Version | Date       | Description                                                                 | Changes Included                                                                                    |
| ------- | ---------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| v1.0.0  | 2025-10-04 | Initial complete working version with full training and inference pipeline. | - Modularized scripts for dataset preparation, model loading, training, inference, and API service. |
|         |            |                                                                             | - Integrated `config.json` for all configuration parameters.                                        |
|         |            |                                                                             | - Added logging functionality for reliability and debugging.                                        |
|         |            |                                                                             | - Support for CPU and GPU inference in a flexible way.                                              |
|         |            |                                                                             | - Configurations for DPO fine-tuning with LoRA on Unsloht Qwen3-0.6B 4-bit model.                   |
|         |            |                                                                             | - Added inference utility with adaptive token size based on VRAM availability.                      |
|         |            |                                                                             | - Added FastAPI server setup (`app.py` and `start_api.py`) for deployment.                          |
|         |            |                                                                             | - Included wandb integration for training tracking.                                                 |
|         |            |                                                                             | - Added README.md documenting configuration choices, workflow, and reasoning.                       |

Versionv 1.0.0:
Codebase State: Stable baseline for fine-tuning Fitness QA bot with Unsloht + LoRA + DPO.
Config State: Finalized config.json, containing dataset paths, training parameters, model configuration, inference settings, wandb settings, and logging configuration.
{
    "train_file": "/kaggle/input/fitness-qa-dpo-based/dpo_train.json",
    "val_file": "/kaggle/input/fitness-qa-dpo-based/dpo_val.json",
    "train_limit": 4200,
    "test_limit": 500,

    "model": {
        "name": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "max_seq_length": 2048,
        "dtype": null,
        "load_in_4bit": true,
        "lora": {
            "r": 16,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "bias": "none",
            "use_gradient_checkpointing": false,
            "random_state": 3407,
            "use_rslora": false,
            "loftq_config": null
        }
    },

    "trainer": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.1,
        "num_train_epochs": 5,
        "learning_rate": 5e-6,
        "logging_steps": 50,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 42,
        "output_dir": "outputs",
        "beta": 0.1,
        "max_length": 1024,
        "max_prompt_length": 256
    },

    "inference": {
        "model_path": "outputs",  
        "max_seq_length": 2048,
        "load_in_4bit": true,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": true,
        "max_new_tokens": 1024,
        "adaptive_generation": true,
        "adaptive_threshold_vram_gb": 10,
        "system_prompt": "You are a helpful AI fitness assistant for gym-goers and vegetarians."
    },

    "wandb": {
        "use": true,
        "project": "fitness-bot",
        "name": "qwen3_0.6b_finetuning"
    },

    "api": {
        "enable": true,
        "host": "0.0.0.0",
        "port": 8000,
        "endpoint": "/predict/",
        "max_request_length": 1024
    },

    "logging": {
        "level": "INFO",
        "log_file": "logs/fitness_bot.log"
    }
}

Key Highlights of This Version:

Fully modularized architecture for maintainability.

Flexible configuration-driven system.

Logging for reliability.

CPU/GPU compatible inference.

Optimized settings for small dataset (~4,200 training + 500 validation samples).

LoRA applied to only QKV and attention modules for efficiency.

wandb tracking enabled.

Production-ready API endpoint configuration.

Results:
| Run Name              | Epochs | Steps | Train Loss | Grad Norm | LR Final | Rewards Accuracy | Rewards Margin | Runtime | Key Insight                                                                                                          |
| --------------------- | -----: | ----: | ---------: | --------: | -------: | ---------------: | -------------: | ------- | -------------------------------------------------------------------------------------------------------------------- |
| qwen3_0.6b_finetuning |      5 |  2625 |     0.0139 |   0.00062 |      0.0 |              1.0 |          12.22 | 1h 47m  | Training saturated early with near‑zero loss from epoch 2. Adjust LR schedule and warm‑up to improve generalization. |

