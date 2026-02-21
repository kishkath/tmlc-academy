import torch
from configurations.config import INFERENCE_CONFIG, USE_WANDB, logger
import wandb

def get_adaptive_max_tokens():
    logger.info("🔍 Checking if adaptive token generation is enabled...")
    if INFERENCE_CONFIG.get("adaptive_generation", False):
        try:
            if torch.cuda.is_available():
                vram_free = torch.cuda.mem_get_info()[0] / 1e9
                logger.info(f"Available VRAM: {vram_free:.2f} GB")
                if vram_free < INFERENCE_CONFIG.get("adaptive_threshold_vram_gb", 10):
                    adjusted_tokens = max(512, int(INFERENCE_CONFIG.get("max_new_tokens", 512) / 2))
                    logger.info(f"Adaptive threshold triggered: reducing max_new_tokens to {adjusted_tokens}")
                    return adjusted_tokens
        except Exception as e:
            logger.warning(f"Adaptive token adjustment failed: {e}")
    max_tokens = INFERENCE_CONFIG.get("max_new_tokens", 512)
    logger.info(f"Using max_new_tokens: {max_tokens}")
    return max_tokens

def predict(model, tokenizer, input_prompt, system_prompt=None):
    logger.info("💬 Starting prediction...")
    logger.info(f"Input prompt: {input_prompt}")

    system_prompt = system_prompt or INFERENCE_CONFIG.get(
        "system_prompt", "You are a helpful assistant."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]
    logger.info(f"System prompt: {system_prompt}")

    try:
        logger.info("🔧 Applying chat template to messages...")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info(f"Formatted text: {text}")

        model_inputs = tokenizer([text], return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        model.to(device)
        model.eval()

        with torch.no_grad():
            max_new_tokens = get_adaptive_max_tokens()
            logger.info(f"Generating text with max_new_tokens={max_new_tokens}...")
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=INFERENCE_CONFIG.get("temperature", 0.7),
                top_p=INFERENCE_CONFIG.get("top_p", 0.9),
                do_sample=INFERENCE_CONFIG.get("do_sample", True)
            )

        logger.info("✅ Generation complete.")
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Generated response: {response}")

        if INFERENCE_CONFIG.get("log_inference", False) and USE_WANDB:
            wandb.log({"inference/max_new_tokens": max_new_tokens, "inference/response_length": len(response.split())})

        return response.strip()

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
