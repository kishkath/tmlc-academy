from core.dataset_preparation import prepare_dataset
from core.model_loader import load_model, load_reference_model
from core.trainer import create_dpo_trainer, train_and_save
from configurations.config import TRAIN_FILE, VAL_FILE, TRAIN_LIMIT, TEST_LIMIT, logger, TRAINER_CONFIG

import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    logger.info(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(TRAIN_FILE, VAL_FILE, TRAIN_LIMIT, TEST_LIMIT)
    logger.info(f"✅ Train size: {len(dataset['train'])}")
    logger.info(f"✅ Test size: {len(dataset['val'])}")
    logger.info(f"✅ Example record: {dataset['train'][0]}")

    logger.info("Loading models...")
    model, tokenizer = load_model()
    ref_model = load_reference_model()
    logger.info("✅ Models loaded successfully")

    logger.info("Setting random seed...")
    set_seed(TRAINER_CONFIG.get("seed", 42))

    logger.info("Creating trainer...")
    trainer = create_dpo_trainer(model, ref_model, tokenizer, dataset)
    logger.info("✅ Trainer created successfully")

    save_path = TRAINER_CONFIG.get("output_dir", "outputs")
    train_and_save(trainer, model, tokenizer, save_path)

