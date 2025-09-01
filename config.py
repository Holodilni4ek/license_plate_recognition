#!/usr/bin/env python3
"""
Configuration Manager
Centralized configuration management for the license plate recognition application.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

@dataclass
class UIConfig:
    """User interface configuration."""
    bg_color: str = "White"
    adm_bg_color: str = "Red"
    window_width: int = 1000
    window_height: int = 800
    min_width: int = 800
    min_height: int = 600

@dataclass
class DateConfig:
    """Date range configuration."""
    min_date: str = "2025-01-01"
    max_date: str = "2025-12-31"

@dataclass
class PathConfig:
    """Path configuration."""
    user_path: str = "Desktop"
    models_path: str = "models"
    plates_path: str = "plates"
    icon_path: str = "docs/app_icon.ico"
    
    @property
    def desktop_path(self) -> str:
        """Get the desktop path."""
        return os.path.join(os.path.expanduser("~"), self.user_path)

@dataclass
class ModelConfig:
    """Model configuration."""
    resnet_model: str = "model_resnet.tflite"
    recognition_model: str = "model_number_recognition.tflite"
    
    # Model download URLs
    resnet_url: str = "https://disk.yandex.ru/d/QavLH1pvpRhLOA"
    recognition_url: str = "https://github.com/sovse/tflite_avto_num_recognation/blob/main/model1_nomer.tflite"
    
    @property
    def resnet_path(self) -> str:
        """Get full path to ResNet model."""
        return os.path.join("models", self.resnet_model)
    
    @property
    def recognition_path(self) -> str:
        """Get full path to recognition model."""
        return os.path.join("models", self.recognition_model)

@dataclass
class ProcessingConfig:
    """Image processing configuration."""
    image_size: int = 1024
    crop_min_ratio: float = 2.0
    rotation_threshold: float = 20.0
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: tuple = (8, 8)
    final_img_width: int = 128
    final_img_height: int = 64

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: str
    dbname: str
    user: str
    password: str
    
    def __post_init__(self):
        """Validate database configuration."""
        if not all([self.host, self.port, self.dbname, self.user, self.password]):
            missing = [k for k, v in self.__dict__.items() if not v]
            raise ValueError(f"Missing database configuration: {missing}")

class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.ui = UIConfig()
        self.dates = DateConfig()
        self.paths = PathConfig()
        self.models = ModelConfig()
        self.processing = ProcessingConfig()
        
        # Database configuration from environment
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", ""),
            port=os.getenv("DB_PORT", ""),
            dbname=os.getenv("DB_NAME", ""),
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", "")
        )
    
    def validate(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Validate paths exist or can be created
            os.makedirs(self.paths.models_path, exist_ok=True)
            os.makedirs(self.paths.plates_path, exist_ok=True)
            
            # Validate database config
            self.database.__post_init__()
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
_config = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
