"""
API endpoints for Instagram content validation.
"""

import asyncio
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlmodel import Session, select, func  # Added select and func imports
from app.database import get_session
from app.models.database import Submission, ContentType, ValidationStatus
from app.models.api import (
    SubmissionRequest,
    SubmissionResponse,
    ValidationDetailResponse,
    AdminStatsResponse,
    HealthCheckResponse
)
from app.services.validator import ValidationService
from app.utils.helpers import save_uploaded_file
from app.utils.exceptions import FileValidationError, ValidationError
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["validation"])

# Initialize validation service
validation_service = ValidationService()

@router.post("/submit", response_model=SubmissionResponse)
async def submit_content(
    instagram_url: str = Form(..., description="Instagram post/story/reel URL"),
    content_type: ContentType = Form(..., description="Type of Instagram content"),
    screenshot: UploadFile = File(..., description="Screenshot of the Instagram content"),
    session: Session = Depends(get_session)
):
    """
    Submit Instagram content for validation.

    This endpoint accepts:
    - Instagram URL (post, story, or reel)
    - Content type specification
    - Screenshot file showing the content

    Returns complete validation results immediately.
    """
    try:
        # Validate file
        if not screenshot.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Save uploaded file
        try:
            screenshot_path = await save_uploaded_file(screenshot)
        except FileValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create submission record
        submission = Submission(
            instagram_url=instagram_url,
            content_type=content_type,
            screenshot_path=screenshot_path,
            validation_status=ValidationStatus.PENDING
        )

        session.add(submission)
        session.commit()
        session.refresh(submission)

        logger.info(f"New submission created: {submission.id}")

        # Run validation immediately (not in background)
        validation_result = await validation_service.validate_submission(submission.id)

        # Get updated submission data
        session.refresh(submission)

        return SubmissionResponse(
            id=submission.id,
            status=submission.validation_status,
            message="ontent validation failed. Some requirements not met",
            created_at=submission.created_at,
            url_username=submission.url_username,
            content_username=submission.content_username,
            username_match=submission.username_match,
            extracted_hashtags=submission.extracted_hashtags,
            hashtags_valid=submission.hashtags_valid,
            missing_hashtags=submission.missing_hashtags,
            is_account_public=submission.is_account_public
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _get_status_message(status: ValidationStatus) -> str:
    """Get appropriate message for validation status."""
    messages = {
        ValidationStatus.VALID: "Content validation successful. All requirements met.",
        ValidationStatus.INVALID: "Content validation failed. Some requirements not met.",
        ValidationStatus.ERROR: "Validation error occurred. Please try again.",
        ValidationStatus.PENDING: "Validation in progress..."
    }
    return messages.get(status, "Unknown status")

@router.get("/submission/{submission_id}", response_model=SubmissionResponse)
async def get_submission_status(
    submission_id: int,
    session: Session = Depends(get_session)
):
    """
    Get the current status and results of a submission.

    Returns detailed validation results if validation is complete.
    """
    submission = session.get(Submission, submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    return SubmissionResponse(
        id=submission.id,
        status=submission.validation_status.value,
        message=_get_status_message(submission),
        created_at=submission.created_at,
        url_username=submission.url_username,
        content_username=submission.content_username,
        username_match=submission.username_match,
        extracted_hashtags=submission.extracted_hashtags,
        hashtags_valid=submission.hashtags_valid,
        missing_hashtags=submission.missing_hashtags,
        is_account_public=submission.is_account_public
    )

@router.get("/submissions", response_model=List[SubmissionResponse])
async def list_submissions(
    skip: int = 0,
    limit: int = 100,
    status: ValidationStatus = None,
    session: Session = Depends(get_session)
):
    """
    List all submissions with optional filtering by status.

    Supports pagination with skip/limit parameters.
    """
    query = select(Submission)

    if status:
        query = query.where(Submission.validation_status == status)

    query = query.offset(skip).limit(limit).order_by(Submission.created_at.desc())

    submissions = session.exec(query).all()

    return [
        SubmissionResponse(
            id=sub.id,
            status=sub.validation_status,
            message=_get_status_message(sub),
            created_at=sub.created_at,
            url_username=sub.url_username,
            content_username=sub.content_username,
            username_match=sub.username_match,
            extracted_hashtags=sub.extracted_hashtags,
            hashtags_valid=sub.hashtags_valid,
            missing_hashtags=sub.missing_hashtags,
            is_account_public=sub.is_account_public
        )
        for sub in submissions
    ]

@router.get("/stats", response_model=AdminStatsResponse)
async def get_validation_stats():
    """
    Get validation statistics.

    Returns counts and success rates for all submissions.
    """
    try:
        stats = await validation_service.get_validation_stats()
        return AdminStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@router.post("/revalidate/{submission_id}")
async def revalidate_submission(
    submission_id: int,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Rerun validation for a specific submission.

    Useful for retrying failed validations or updating results.
    """
    submission = session.get(Submission, submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Reset status to pending
    submission.validation_status = ValidationStatus.PENDING
    submission.error_message = None
    submission.validated_at = None

    session.add(submission)
    session.commit()

    # Start validation in background
    # background_tasks.add_task(
    #     run_validation_task,
    #     submission_id
    # )

    return {"message": "Revalidation started", "submission_id": submission_id}

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(session: Session = Depends(get_session)):
    """
    Health check endpoint.

    Verifies database connectivity and configuration.
    """
    try:
        # Test database connection
        session.exec(select(1)).first()
        db_connected = True
    except Exception:
        db_connected = False

    # Check OpenAI configuration
    openai_configured = bool(settings.openai_api_key)

    return HealthCheckResponse(
        status="healthy" if db_connected and openai_configured else "unhealthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        database_connected=db_connected,
        openai_configured=openai_configured
    )

def _get_status_message(submission: Submission) -> str:
    """Generate status message for submission."""
    if submission.validation_status == ValidationStatus.PENDING:
        return "Validation in progress..."
    elif submission.validation_status == ValidationStatus.VALID:
        return "Submission is valid and meets all requirements."
    elif submission.validation_status == ValidationStatus.INVALID:
        reasons = []
        if submission.username_match is False:
            reasons.append("Username mismatch between URL and content")
        if submission.hashtags_valid is False:
            missing = submission.missing_hashtags or []
            reasons.append(f"Missing required hashtags: {', '.join(missing)}")

        if reasons:
            return f"Submission is invalid: {'; '.join(reasons)}"
        else:
            return "Submission is invalid."
    elif submission.validation_status == ValidationStatus.ERROR:
        return f"Validation error: {submission.error_message or 'Unknown error'}"

    return "Unknown status"

# Note: Exception handlers are moved to main.py since APIRouter doesn't support them