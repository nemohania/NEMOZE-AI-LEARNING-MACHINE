"""
Configuration settings for the application
"""

from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Backend settings
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/ai_machine"
    )
    
    # JWT settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML settings
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "transformer")
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 0.001))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    EPOCHS: int = int(os.getenv("EPOCHS", 10))
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
    ]
    
    # Data settings
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    LOGS_DIR: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
