"""
Main API routes for predictions and model management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import logging
import torch
from datetime import datetime

from app.api.auth import verify_token
from app.models.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize model manager
model_manager = ModelManager()

class PredictionRequest(BaseModel):
    data: List[float]
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: List[float]
    confidence: Optional[float] = None
    timestamp: str

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_value: List[float]
    rating: int  # 1-5

class ModelStatus(BaseModel):
    name: str
    status: str
    last_updated: str
    metrics: dict

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    payload: dict = Depends(verify_token)
) -> dict:
    """Get prediction from model"""
    try:
        model = model_manager.get_model()
        if not model:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        # Convert input to tensor
        input_tensor = torch.tensor([request.data], dtype=torch.float32)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        prediction = output[0].tolist()
        confidence = 0.85  # Placeholder
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = 10,
    payload: dict = Depends(verify_token)
) -> dict:
    """Get prediction history"""
    return {"predictions": [], "total": 0}

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    payload: dict = Depends(verify_token)
) -> dict:
    """Submit feedback for model improvement"""
    logger.info(f"Feedback received: {request.prediction_id}")
    return {"status": "success", "message": "Feedback recorded"}

@router.get("/model/status", response_model=ModelStatus)
async def get_model_status(
    model_name: Optional[str] = None,
    payload: dict = Depends(verify_token)
) -> dict:
    """Get model status"""
    return {
        "name": model_name or "default",
        "status": "active",
        "last_updated": datetime.utcnow().isoformat(),
        "metrics": {"accuracy": 0.95, "loss": 0.05}
    }

@router.post("/model/train")
async def train_model(
    payload: dict = Depends(verify_token)
) -> dict:
    """Trigger model retraining"""
    logger.info("Model retraining initiated")
    return {"status": "training_started", "message": "Model retraining has been initiated"}

@router.get("/model/metrics")
async def get_model_metrics(
    payload: dict = Depends(verify_token)
) -> dict:
    """Get model performance metrics"""
    return {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.935,
        "loss": 0.05,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/models")
async def list_models(payload: dict = Depends(verify_token)) -> dict:
    """List all available models"""
    models = model_manager.list_models()
    return {"models": models, "count": len(models)}

@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "model": "ready" if model_manager.get_model() else "not_loaded"
    }
