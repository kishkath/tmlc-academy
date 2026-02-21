from configurations.config import INFERENCE_MODEL_PATH, INFERENCE_MAX_SEQ_LENGTH, INFERENCE_LOAD_IN_4BIT, logger
from core.model_loader import load_finetuned_model
from core.inference_utils import predict

if __name__ == "__main__":
    logger.info(f"🔍 Loading fine-tuned model from '{INFERENCE_MODEL_PATH}'")
    try:
        model, tokenizer = load_finetuned_model(
            INFERENCE_MODEL_PATH, INFERENCE_MAX_SEQ_LENGTH, INFERENCE_LOAD_IN_4BIT
        )
        logger.info(f"✅ Fine-tuned model loaded successfully from '{INFERENCE_MODEL_PATH}'")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    input_prompt = "Common mistakes in micronutrients (iron, calcium, vitamin D) for vegetarian lifters?"
    logger.info(f"💬 Running inference for prompt: {input_prompt}")

    try:
        response = predict(model, tokenizer, input_prompt)
        logger.info(f"✅ Prediction completed")
        print("\n💡 Prediction:\n", response)
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
