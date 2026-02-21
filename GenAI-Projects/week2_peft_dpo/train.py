import logging
import sys
import torch
from pathlib import Path
import os
from tqdm.auto import tqdm
import wandb

# ------------------- Paths -------------------
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)

# ------------------- Imports -------------------
from configurations.config import (
    DEVICE,
    N_GPUS,
    MODEL_NAME,
    TRAIN_FILE,
    EVAL_FILE,
    MAX_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_TARGET_MODULES,
    LORA_DROPOUT,
    LORA_BIAS,
    LORA_TASK_TYPE,
    TRAINING_ARGS,
    QUESTIONS,
)
from dataset.data_utils import create_dataset, tokenize_dataset
from finetuning.model import load_model_and_tokenizer
from finetuning.trainer import get_lora_config, get_training_args, get_trainer

# ------------------- Logging -------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def log_print(message):
    logger.info(message)
    print(message)

# ------------------- Weights & Biases Init -------------------
wandb.init(
    project="fitness-qa-bot",
    config=TRAINING_ARGS,
    name="qwen2.5-finetuning"
)

# ------------------- Model & Tokenizer -------------------
log_print(f"Loading model and tokenizer: {MODEL_NAME}")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE)
log_print("Model and tokenizer loaded.")

# ------------------- Dataset -------------------
log_print(f"Loading datasets: train={TRAIN_FILE}, eval={EVAL_FILE}")
train_dataset, eval_dataset = create_dataset(TRAIN_FILE, EVAL_FILE)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=MAX_LENGTH)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=MAX_LENGTH)
log_print("Datasets tokenized.")

# ------------------- LoRA & Training -------------------
log_print("Creating LoRA config and training arguments...")
lora_config = get_lora_config(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type=LORA_TASK_TYPE
)
training_args = get_training_args(TRAINING_ARGS)
training_args.remove_unused_columns = False
training_args.report_to = "none"  # keep console + wandb
training_args.logging_steps = TRAINING_ARGS.get("logging_steps", 1)
# training_args.load_best_model_at_end = TRAINING_ARGS.get("load_best_model_at_end", True)

log_print("LoRA config and training arguments ready.")

trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)
log_print("Trainer initialized.")

# ------------------- Training -------------------
log_print("Starting training...")

num_train_steps = len(tokenized_train_dataset) // training_args.per_device_train_batch_size
pbar = tqdm(total=num_train_steps, desc="Training Progress", unit="step", ncols=100)

for _ in trainer.train():
    pbar.update(1)

    if trainer.state.global_step % training_args.logging_steps == 0:
        if trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            current_loss = last_log.get("loss", last_log.get("eval_loss", None))
            lr = last_log.get("learning_rate")
            grad_norm = last_log.get("grad_norm")
            token_acc = last_log.get("mean_token_accuracy")
            entropy = last_log.get("entropy")
        else:
            current_loss, lr, grad_norm, token_acc, entropy = None, None, None, None, None

        log_print(
            f"Step {trainer.state.global_step}/{num_train_steps} "
            f"- Loss: {current_loss}, LR: {lr}, GradNorm: {grad_norm}, "
            f"TokenAcc: {token_acc}, Entropy: {entropy}"
        )

        # Log metrics to wandb
        wandb.log({
            "step": trainer.state.global_step,
            "loss": current_loss,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "token_accuracy": token_acc,
            "entropy": entropy
        })

        # ---- Sample Generations Logging ----
        if trainer.state.global_step % (training_args.logging_steps * 5) == 0:
            wandb_table = wandb.Table(columns=["Question", "Generated Answer"])
            model.eval()
            for q in QUESTIONS[:2]:  # just take 2 example questions
                inputs = tokenizer(q, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7
                    )
                answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                wandb_table.add_data(q, answer)
            wandb.log({"sample_generations": wandb_table})

pbar.close()
log_print("Training completed.")

# ------------------- Save -------------------
log_print("Saving LoRA adapter & tokenizer...")
trainer.model.save_pretrained(TRAINING_ARGS.get("output_dir"))
tokenizer.save_pretrained(TRAINING_ARGS.get("output_dir"))
log_print(f"All artifacts saved to {TRAINING_ARGS.get('output_dir')}")

wandb.finish()


