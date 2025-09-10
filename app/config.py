"""
Configuration management for the Instagram Validator application.
Handles environment variables and application settings.
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Admin Panel Configuration
    admin_username: str = os.getenv("ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "admin123")
    admin_secret_key: str = os.getenv("ADMIN_SECRET_KEY", "your-secret-key-change-in-production")

    # Campaign Configuration
    required_hashtags: str = os.getenv("REQUIRED_HASHTAGS", "#MyBrand,#MyCampaign")

    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./instagram_validator.db")

    # Application Configuration
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    upload_dir: str = "uploads"

    @property
    def required_hashtags_list(self) -> List[str]:
        """Convert comma-separated hashtags string to list."""
        return [tag.strip() for tag in self.required_hashtags.split(",") if tag.strip()]

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()

# Validate required settings
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Create upload directory if it doesn't exist
os.makedirs(settings.upload_dir, exist_ok=True)