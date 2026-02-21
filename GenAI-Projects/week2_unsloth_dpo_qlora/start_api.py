from app import app
from configurations.config import API_CONFIG, logger
import uvicorn

if __name__ == "__main__":
    logger.info(f"Launching Fitness QA Bot API at {API_CONFIG.get('host')}:{API_CONFIG.get('port')}")
    uvicorn.run(
        "app:app",
        host=API_CONFIG.get("host", "0.0.0.0"),
        port=API_CONFIG.get("port", 8000),
        reload=True
    )
