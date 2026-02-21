from trl import DPOTrainer, DPOConfig
import wandb
import torch
from configurations.config import TRAINER_CONFIG, USE_WANDB, WANDB_PROJECT, WANDB_NAME, logger


def create_dpo_trainer(model, ref_model, tokenizer, dataset):
    logger.info("Creating DPO Trainer...")

    dpo_args = DPOConfig(
        per_device_train_batch_size=TRAINER_CONFIG.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=TRAINER_CONFIG.get("gradient_accumulation_steps", 4),
        warmup_ratio=TRAINER_CONFIG.get("warmup_ratio", 0.2),
        num_train_epochs=TRAINER_CONFIG.get("num_train_epochs", 4),
        learning_rate=TRAINER_CONFIG.get("learning_rate", 3e-5),
        logging_steps=TRAINER_CONFIG.get("logging_steps", 50),
        optim=TRAINER_CONFIG.get("optim", "adamw_8bit"),
        weight_decay=TRAINER_CONFIG.get("weight_decay", 0.01),
        lr_scheduler_type=TRAINER_CONFIG.get("lr_scheduler_type", "cosine"),
        seed=TRAINER_CONFIG.get("seed", 42),
        output_dir=TRAINER_CONFIG.get("output_dir", "outputs"),
        report_to="wandb" if USE_WANDB else "none"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        beta=TRAINER_CONFIG.get("beta", 0.1),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],  # ✅ fixed: use val dataset for validation
        tokenizer=tokenizer,
        max_length=TRAINER_CONFIG.get("max_length", 1024),
        max_prompt_length=TRAINER_CONFIG.get("max_prompt_length", 256),
    )

    if USE_WANDB:
        logger.info(f"Initializing WandB project: {WANDB_PROJECT}, run name: {WANDB_NAME}")
        wandb.init(project=WANDB_PROJECT, config=TRAINER_CONFIG, name=WANDB_NAME)

    logger.info("✅ DPO Trainer created successfully.")
    return trainer


def train_and_save(trainer, model, tokenizer, save_path=None):
    logger.info("🚀 Starting training...")

    num_epochs = TRAINER_CONFIG.get("num_train_epochs", 4)
    eval_steps = TRAINER_CONFIG.get("eval_steps", 400)

    for epoch in range(num_epochs):
        logger.info(f"🔹 Epoch {epoch + 1}/{num_epochs}")

        trainer.train()

        # Validation at defined intervals
        if eval_steps > 0 and trainer.state.global_step % eval_steps == 0:
            logger.info(f"🧪 Running validation at step {trainer.state.global_step}...")
            eval_metrics = trainer.evaluate()

            logger.info(f"📊 Validation metrics at step {trainer.state.global_step}: {eval_metrics}")

            if USE_WANDB:
                wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()}, step=trainer.state.global_step)

    # Final validation at the end of training
    logger.info("🧪 Running final validation...")
    eval_metrics = trainer.evaluate()
    logger.info(f"📊 Final validation metrics: {eval_metrics}")

    if USE_WANDB:
        wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()}, step=trainer.state.global_step)
        wandb.finish()

    save_path = save_path or TRAINER_CONFIG.get("output_dir", "outputs")
    logger.info(f"💾 Saving model to {save_path}...")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("✅ Model saved successfully.")
