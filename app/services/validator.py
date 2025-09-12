"""
Main validation service that orchestrates the validation workflow.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sqlmodel import Session, select, func
from app.database import get_session
from app.models.database import Submission, ValidationResult, ValidationStatus, ContentType
from app.models.api import ExtractedData
from app.services.instagram_scraper import InstagramScraper
from app.services.ocr_service import OCRService
from app.services.openai_analyzer import OpenAIAnalyzer
from app.config import settings
from app.utils.exceptions import ValidationError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationService:
    """Main service for validating Instagram submissions."""

    def __init__(self):
        """Initialize validation service."""
        self.scraper = InstagramScraper()
        self.ocr_service = OCRService()
        self.analyzer = OpenAIAnalyzer()

    async def validate_submission(self, submission_id: int) -> Dict[str, Any]:
        """
        Perform complete validation workflow for a submission.

        Args:
            submission_id: ID of the submission to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        session_gen = get_session()
        session = next(session_gen)

        try:
            # Get submission from database
            submission = session.get(Submission, submission_id)
            if not submission:
                raise ValidationError(f"Submission {submission_id} not found")

            logger.info(f"Starting validation for submission {submission_id}")

            # Create validation result record
            validation_result = ValidationResult(submission_id=submission_id)

            # Step 1: Extract content data (try instaloader -> OCR -> LLM)
            extracted_data, extraction_success = await self._extract_content_data(
                submission, validation_result
            )

            if extracted_data:
                submission.content_username = extracted_data.username
                submission.extracted_hashtags = extracted_data.hashtags
                submission.extracted_data = {
                    "content_type": extracted_data.content_type.value if extracted_data.content_type else None,
                    "confidence": extracted_data.confidence,
                    "extraction_method": extracted_data.extraction_method
                }

            # Step 2: Validate hashtags
            hashtags_valid, missing_hashtags, hashtag_validation_success = self._validate_hashtags(
                extracted_data.hashtags if extracted_data else [], validation_result
            )
            submission.hashtags_valid = hashtags_valid
            submission.missing_hashtags = missing_hashtags

            # Determine overall validation status
            overall_success = extraction_success and hashtag_validation_success

            if overall_success and hashtags_valid:
                submission.validation_status = ValidationStatus.VALID
            elif overall_success:
                submission.validation_status = ValidationStatus.INVALID
            else:
                submission.validation_status = ValidationStatus.ERROR

            submission.validated_at = datetime.utcnow()

            # Save results
            session.add(validation_result)
            session.add(submission)
            session.commit()

            logger.info(f"Validation completed for submission {submission_id}: {submission.validation_status}")

            return {
                "submission_id": submission_id,
                "status": submission.validation_status,
                "hashtags_valid": hashtags_valid,
                "missing_hashtags": missing_hashtags,
                "extraction_method": extracted_data.extraction_method if extracted_data else None,
                "confidence": extracted_data.confidence if extracted_data else None,
                "username": extracted_data.username if extracted_data else None
            }

        except Exception as e:
            # Update submission with error
            submission.validation_status = ValidationStatus.ERROR
            submission.error_message = str(e)
            submission.validated_at = datetime.utcnow()
            session.add(submission)
            session.commit()

            logger.error(f"Validation failed for submission {submission_id}: {str(e)}")
            raise ValidationError(f"Validation failed: {str(e)}")

        finally:
            session.close()

    async def _extract_content_data(self, submission: Submission,
                                    validation_result: ValidationResult) -> Tuple[Optional[ExtractedData], bool]:
        """
        Extract content data using three-tier approach: Instaloader -> OCR -> LLM.
        """
        try:
            extracted_data = None

            # Step 1: Try Instaloader method first
            try:
                extracted_data = await self.scraper.scrape_post_data(submission.instagram_url)
                if extracted_data:
                    validation_result.extraction_method = "instaloader_shortcode"
                    validation_result.raw_scraped_data = {
                        "username": extracted_data.username,
                        "hashtags": extracted_data.hashtags,
                        "content_type": extracted_data.content_type.value if extracted_data.content_type else None
                    }
                    logger.info("Successfully extracted data via Instaloader shortcode method")
                    validation_result.content_extraction_success = True
                    validation_result.extraction_confidence = extracted_data.confidence
                    return extracted_data, True
            except Exception as e:
                logger.warning(f"Instaloader extraction failed: {str(e)}")

            # Step 2: Try OCR method if Instaloader failed
            try:
                extracted_data = await self.ocr_service.analyze_screenshot_with_ocr(
                    submission.screenshot_path,
                    submission.content_type
                )
                if extracted_data and (extracted_data.hashtags or extracted_data.username):
                    validation_result.extraction_method = "ocr"
                    validation_result.raw_ocr_response = {
                        "username": extracted_data.username,
                        "hashtags": extracted_data.hashtags,
                        "content_type": extracted_data.content_type.value if extracted_data.content_type else None
                    }
                    logger.info(f"Successfully extracted data via OCR method. Found {len(extracted_data.hashtags)} hashtags")
                    validation_result.content_extraction_success = True
                    validation_result.extraction_confidence = extracted_data.confidence
                    return extracted_data, True
                else:
                    logger.info("OCR method found no useful data")
            except Exception as e:
                logger.warning(f"OCR extraction failed: {str(e)}")

            # Step 3: Fall back to LLM analysis if both Instaloader and OCR failed
            try:
                extracted_data = await self.analyzer.analyze_screenshot(
                    submission.screenshot_path,
                    submission.content_type
                )
                validation_result.extraction_method = "llm"
                validation_result.raw_llm_response = {
                    "username": extracted_data.username,
                    "hashtags": extracted_data.hashtags,
                    "content_type": extracted_data.content_type.value if extracted_data.content_type else None,
                    "confidence": extracted_data.confidence
                }
                logger.info("Successfully extracted data via LLM analysis")
                validation_result.content_extraction_success = True
                validation_result.extraction_confidence = extracted_data.confidence
                return extracted_data, True
            except Exception as e:
                logger.error(f"LLM extraction also failed: {str(e)}")

            # If all methods failed
            logger.error("All extraction methods (Instaloader, OCR, LLM) failed")
            validation_result.content_extraction_success = False
            return None, False

        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            validation_result.content_extraction_success = False
            return None, False

    def _validate_hashtags(self, extracted_hashtags: List[str], validation_result: ValidationResult) -> Tuple[
        bool, List[str], bool]:
        """Validate that required hashtags are present."""
        try:
            required_hashtags = settings.required_hashtags_list
            if not required_hashtags:
                # No required hashtags configured
                validation_result.hashtag_validation_success = True
                return True, [], True

            # Normalize hashtags for comparison
            extracted_normalized = [tag.lower().strip() for tag in extracted_hashtags]
            required_normalized = [tag.lower().strip() for tag in required_hashtags]

            # Find missing hashtags
            missing_hashtags = []
            for required_tag in required_normalized:
                if required_tag not in extracted_normalized:
                    missing_hashtags.append(required_tag)

            hashtags_valid = len(missing_hashtags) == 0
            validation_result.hashtag_validation_success = True

            return hashtags_valid, missing_hashtags, True

        except Exception as e:
            logger.error(f"Hashtag validation failed: {str(e)}")
            validation_result.hashtag_validation_success = False
            return False, [], False

    async def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        session_gen = get_session()
        session = next(session_gen)

        try:
            # Count submissions by status
            total_count = session.exec(select(func.count(Submission.id))).first()
            valid_count = session.exec(
                select(func.count(Submission.id)).where(Submission.validation_status == ValidationStatus.VALID)
            ).first()
            invalid_count = session.exec(
                select(func.count(Submission.id)).where(Submission.validation_status == ValidationStatus.INVALID)
            ).first()
            pending_count = session.exec(
                select(func.count(Submission.id)).where(Submission.validation_status == ValidationStatus.PENDING)
            ).first()
            error_count = session.exec(
                select(func.count(Submission.id)).where(Submission.validation_status == ValidationStatus.ERROR)
            ).first()

            # Calculate success rate
            processed_count = total_count - pending_count
            success_rate = (valid_count / processed_count * 100) if processed_count > 0 else 0

            # Get extraction method statistics
            extraction_stats = await self._get_extraction_method_stats(session)

            return {
                "total_submissions": total_count,
                "valid_submissions": valid_count,
                "invalid_submissions": invalid_count,
                "pending_submissions": pending_count,
                "error_submissions": error_count,
                "success_rate": round(success_rate, 2),
                "extraction_methods": extraction_stats
            }

        finally:
            session.close()

    async def _get_extraction_method_stats(self, session: Session) -> Dict[str, int]:
        """Get statistics on extraction methods used."""
        try:
            # This would require adding extraction_method field to Submission model
            # For now, return placeholder data
            return {
                "instaloader": 0,
                "ocr": 0,
                "llm": 0
            }
        except Exception as e:
            logger.warning(f"Failed to get extraction method stats: {str(e)}")
            return {}