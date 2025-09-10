"""
Database models using SQLModel.
"""

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column, JSON
from enum import Enum


class ContentType(str, Enum):
    """Content type enumeration."""
    POST = "post"
    STORY = "story"
    REELS = "reels"


class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"


class Submission(SQLModel, table=True):
    """Database model for Instagram submissions."""

    id: Optional[int] = Field(default=None, primary_key=True)

    # User provided data
    instagram_url: str = Field(index=True)
    content_type: ContentType
    screenshot_path: str

    # Extracted data
    url_username: Optional[str] = Field(default=None, index=True)
    content_username: Optional[str] = Field(default=None, index=True)
    extracted_hashtags: Optional[list] = Field(default=None, sa_column=Column(JSON))
    is_account_public: Optional[bool] = Field(default=None)

    # Validation results
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING)
    username_match: Optional[bool] = Field(default=None)
    hashtags_valid: Optional[bool] = Field(default=None)
    missing_hashtags: Optional[list] = Field(default=None, sa_column=Column(JSON))

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)

    # Additional extracted data
    extracted_data: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class ValidationResult(SQLModel, table=True):
    """Database model for detailed validation results."""

    id: Optional[int] = Field(default=None, primary_key=True)
    submission_id: int = Field(foreign_key="submission.id", index=True)

    # Validation steps
    url_parsing_success: bool = Field(default=False)
    account_check_success: bool = Field(default=False)
    content_extraction_success: bool = Field(default=False)
    username_validation_success: bool = Field(default=False)
    hashtag_validation_success: bool = Field(default=False)

    # Detailed results
    extraction_method: Optional[str] = Field(default=None)  # "scraping" or "llm"
    extraction_confidence: Optional[float] = Field(default=None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Raw data for debugging
    raw_scraped_data: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    raw_llm_response: Optional[dict] = Field(default=None, sa_column=Column(JSON))