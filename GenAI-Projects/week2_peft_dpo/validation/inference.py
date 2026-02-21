import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from configurations.test_config import TestConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Load inference configuration
cfg = TestConfig("configurations/inference_configs.json")


def save_merged_model(base_model_name, adapter_path, save_path):
    logger.info("Loading base model for merging...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")

    logger.info("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)
    logger.info("Merged model saved successfully.")


def run_inference(model_path, questions, generation_params, mode, base_model_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if mode == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    faq_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for question in questions:
        logger.info(f"Question: {question}")
        response = faq_pipeline(question, **generation_params)
        logger.info(f"Answer: {response[0]['generated_text']}\n")


# ------------------- Main -------------------
if __name__ == "__main__":
    if cfg.mode == "merged":
        save_merged_model(cfg.base_model_name, cfg.adapter_path, cfg.merged_model_path)
        run_inference(cfg.merged_model_path, cfg.questions, cfg.generation, cfg.mode, cfg.base_model_name)
    else:
        run_inference(cfg.adapter_path, cfg.questions, cfg.generation, cfg.mode, cfg.base_model_name)



