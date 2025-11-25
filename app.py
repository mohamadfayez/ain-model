"""
FastAPI Server for AIN Model on Cloud Run
Complete working version ready for deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AIN Model API", version="1.0.0")

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

# Request model
class InferRequest(BaseModel):
    messages: List[Dict[str, Any]]
    max_new_tokens: int = 128

# Response model
class InferResponse(BaseModel):
    result: str
    status: str = "success"


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, processor
    
    try:
        logger.info("Loading AIN model...")
        
        # Load model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "MBZUAI/AIN",
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained("MBZUAI/AIN")
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "AIN-7B",
        "endpoints": {
            "infer": "/infer (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "device": str(next(model.parameters()).device) if model else "unknown"
    }


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Main inference endpoint
    
    Expected format:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data:image/jpeg;base64,..."},
                    {"type": "text", "text": "Your prompt here"}
                ]
            }
        ],
        "max_new_tokens": 128
    }
    """
    global model, processor
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        messages = request.messages
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=request.max_new_tokens
            )
        
        # Trim and decode
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
        
        return InferResponse(result=result)
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
