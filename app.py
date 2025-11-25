"""
FastAPI Server for AIN Model on Cloud Run
Optimized version with delayed imports
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AIN Model API",
    version="1.0.0",
    description="Arabic Inclusive Multimodal Model API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and processor
model = None
processor = None
model_loading = False

# Request model
class InferRequest(BaseModel):
    messages: List[Dict[str, Any]]
    max_new_tokens: int = 128

# Response model
class InferResponse(BaseModel):
    result: str
    status: str = "success"


def load_model_sync():
    """Load model synchronously with proper error handling"""
    global model, processor, model_loading
    
    if model is not None:
        logger.info("Model already loaded")
        return True
    
    if model_loading:
        logger.info("Model is currently loading")
        return False
    
    try:
        model_loading = True
        logger.info("="*60)
        logger.info("Starting model loading process...")
        logger.info("="*60)
        
        # Import heavy libraries only when needed
        logger.info("Importing transformers...")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        logger.info("Importing torch...")
        import torch
        
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Load model
        logger.info("Loading model from MBZUAI/AIN...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "MBZUAI/AIN",
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        logger.info("Model loaded successfully!")
        
        # Load processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained("MBZUAI/AIN")
        logger.info("Processor loaded successfully!")
        
        logger.info("="*60)
        logger.info("Model loading complete!")
        logger.info("="*60)
        
        model_loading = False
        return True
        
    except Exception as e:
        logger.error("="*60)
        logger.error(f"Error loading model: {e}")
        logger.error("="*60)
        import traceback
        logger.error(traceback.format_exc())
        model_loading = False
        return False


@app.get("/")
async def root():
    """Health check endpoint - responds immediately"""
    return {
        "status": "running",
        "service": "AIN Model API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "endpoints": {
            "health": "/health (GET)",
            "load": "/load (POST)",
            "infer": "/infer (POST)"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "model_loading": model_loading,
    }
    
    if model is not None:
        try:
            import torch
            device = next(model.parameters()).device
            health_status["device"] = str(device)
            health_status["cuda_available"] = torch.cuda.is_available()
        except Exception as e:
            health_status["device_error"] = str(e)
    
    return health_status


@app.post("/load")
async def load():
    """Manually trigger model loading"""
    logger.info("Received request to load model")
    
    if model is not None:
        return {"status": "already_loaded"}
    
    if model_loading:
        return {"status": "loading"}
    
    try:
        success = load_model_sync()
        return {"status": "loaded" if success else "loading"}
    except Exception as e:
        logger.error(f"Error in /load endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """Main inference endpoint"""
    global model, processor
    
    logger.info("Received inference request")
    
    # Load model on first request if not loaded
    if model is None and not model_loading:
        logger.info("Model not loaded, loading on first request...")
        load_model_sync()
    
    # Wait if model is loading
    if model_loading:
        logger.info("Waiting for model to load...")
        import time
        wait_time = 0
        max_wait = 600
        
        while model_loading and wait_time < max_wait:
            time.sleep(2)
            wait_time += 2
            if wait_time % 30 == 0:
                logger.info(f"Still waiting... ({wait_time}s)")
        
        if model_loading:
            raise HTTPException(status_code=503, detail="Model loading timeout")
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info("Processing inference...")
        
        from qwen_vl_utils import process_vision_info
        import torch
        
        messages = request.messages
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        result = output_text[0] if output_text else ""
        
        logger.info(f"Inference complete ({len(result)} chars)")
        
        return InferResponse(result=result)
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("="*60)
    logger.info("AIN Model API Starting Up")
    logger.info(f"Port: {os.environ.get('PORT', '8080')}")
    logger.info("Model will be loaded on first request (lazy loading)")
    logger.info("="*60)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
