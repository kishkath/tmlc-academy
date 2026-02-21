from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from configurations.config import CONFIG, API_CONFIG, logger
from core.model_loader import load_finetuned_model
from core.inference_utils import predict
import uvicorn

app = FastAPI(
    title="Fitness QA Bot API",
    version="1.0",
    description="Qwen3-0.6B Finetuned Model for Fitness QA"
)

# ----------------------------- Request Schema -----------------------------
class QueryRequest(BaseModel):
    user_query: str
    max_new_tokens: int = 256


# ----------------------------- Load Model -----------------------------
logger.info("🔍 Loading fine-tuned model...")
try:
    model, tokenizer = load_finetuned_model(
        CONFIG["inference"]["model_path"],
        CONFIG["inference"]["max_seq_length"],
        CONFIG["inference"]["load_in_4bit"]
    )
    logger.info(f"✅ Model loaded successfully from '{CONFIG['inference']['model_path']}'")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise e

model.eval()


# ----------------------------- API Routes -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Fitness QA Bot API (Qwen3-0.6B Finetuned Model)"}


@app.post(API_CONFIG.get("endpoint", "/predict/"))
def predict_endpoint(req: QueryRequest):
    try:
        logger.info(f"Received request: {req.user_query}")
        response = predict(
            model,
            tokenizer,
            req.user_query
        )
        logger.info("✅ Prediction completed")
        return {"response": response.strip()}
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------- Run API -----------------------------
if __name__ == "__main__":
    host = API_CONFIG.get("host", "0.0.0.0")
    port = API_CONFIG.get("port", 8000)
    logger.info(f"🚀 Starting API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
