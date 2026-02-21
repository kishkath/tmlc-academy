import logging
import os
import json
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from configurations.config import (
    MODE,
    BASE_MODEL_NAME,
    ADAPTER_PATH,
    MERGED_MODEL_PATH,
    GENERATION_PARAMS,
    QUESTIONS
)

# ------------------- Paths -------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def log_print(message):
    logger.info(message)
    print(message)

# ------------------- Weights & Biases Init -------------------
wandb.init(
    project="fitness-qa-bot",
    name="qwen-finetuned-inference",
    job_type="inference"
)

# ------------------- Functions -------------------
def save_merged_model(base_model_name, adapter_path, save_path):
    log_print("Loading base model for merging...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
    log_print("Base model loaded.")

    log_print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    log_print("LoRA adapter applied.")

    log_print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    log_print("LoRA weights merged.")

    log_print(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)
    log_print("Merged model saved successfully.")


def run_inference(model_path, questions, generation_params, mode, base_model_name=None):
    log_print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    log_print("Tokenizer loaded.")

    if mode == "lora":
        log_print("Loading base model for LoRA inference...")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
        log_print("Base model loaded. Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
        log_print("LoRA adapter applied.")
    else:
        log_print("Loading full model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        log_print("Model loaded.")

    log_print("Initializing text-generation pipeline...")
    faq_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    log_print("Pipeline ready. Starting inference...")

    wandb_table = wandb.Table(columns=["Question", "Generated Answer"])
    results = []

    for question in questions:
        log_print(f"Question: {question}")
        response = faq_pipeline(question, **generation_params)
        answer = response[0]["generated_text"]
        log_print(f"Answer: {answer}\n")

        # Log to wandb table
        wandb_table.add_data(question, answer)

        # Save to results list for JSON output
        results.append({"question": question, "answer": answer})

    # Log results to wandb
    wandb.log({"inference_results": wandb_table})

    # Save results locally as JSON
    output_path = os.path.join(REPO_ROOT, "outputs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    log_print(f"Inference results saved to {output_path}")


# ------------------- Run -------------------
if MODE == "merged":
    log_print("Mode set to 'merged'. Merging LoRA adapter...")
    save_merged_model(BASE_MODEL_NAME, ADAPTER_PATH, MERGED_MODEL_PATH)
    run_inference(MERGED_MODEL_PATH, QUESTIONS, GENERATION_PARAMS, MODE, BASE_MODEL_NAME)
else:
    log_print("Mode set to 'lora'. Running inference with LoRA adapter...")
    run_inference(ADAPTER_PATH, QUESTIONS, GENERATION_PARAMS, MODE, BASE_MODEL_NAME)

log_print("Inference completed.")
wandb.finish()

