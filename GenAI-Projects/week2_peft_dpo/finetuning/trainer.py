from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def get_lora_config(r, lora_alpha, target_modules, lora_dropout, bias, task_type):
    """
    Accept individual LoRA parameters and return a LoraConfig object
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )

def get_training_args(cfg):
    """
    Convert training configuration dictionary to transformers TrainingArguments
    """
    return TrainingArguments(
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 5e-5),
        warmup_steps=cfg.get("warmup_steps", 40),
        num_train_epochs=cfg.get("num_train_epochs", 4),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 2),
        output_dir=cfg.get("output_dir", "./output"),
        logging_dir=cfg.get("logging_dir", "./logs"),
        report_to=cfg.get("report_to", "none")
    )

def get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args):
    """
    Return an SFTTrainer for fine-tuning
    """
    return SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        peft_config=lora_config,
        args=training_args
    )
