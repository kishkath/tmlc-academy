import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name, device="cpu"):
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer
