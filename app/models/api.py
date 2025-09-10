"""
API models using Pydantic for request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from .database import ContentType, ValidationStatus


class ExtractedData(BaseModel):
    """Model for extracted content data."""
    username: Optional[str] = None
    hashtags: List[str] = []
    content_type: Optional[ContentType] = None
    confidence: Optional[float] = None
    extraction_method: Optional[str] = None


class SubmissionRequest(BaseModel):
    """Request model for content submission."""
    instagram_url: HttpUrl = Field(..., description="Instagram post/story/reel URL")
    content_type: ContentType = Field(..., description="Type of Instagram content")

    class Config:
        json_schema_extra = {
            "example": {
                "instagram_url": "https://www.instagram.com/p/CaBC123def/",
                "content_type": "post"
            }
        }


class SubmissionResponse(BaseModel):
    """Response model for content submission."""
    id: int
    status: ValidationStatus
    message: str
    created_at: datetime

    # Validation details (if completed)
    url_username: Optional[str] = None
    content_username: Optional[str] = None
    username_match: Optional[bool] = None
    extracted_hashtags: Optional[List[str]] = None
    hashtags_valid: Optional[bool] = None
    missing_hashtags: Optional[List[str]] = None
    is_account_public: Optional[bool] = None

    class Config:
        from_attributes = True


class ValidationDetailResponse(BaseModel):
    """Detailed validation response."""
    submission_id: int
    validation_status: ValidationStatus
    steps_completed: Dict[str, bool]
    extracted_data: Optional[ExtractedData] = None
    validation_errors: List[str] = []
    created_at: datetime
    validated_at: Optional[datetime] = None


class AdminStatsResponse(BaseModel):
    """Admin statistics response."""
    total_submissions: int
    valid_submissions: int
    invalid_submissions: int
    pending_submissions: int
    error_submissions: int
    success_rate: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    database_connected: bool
    openai_configured: bool