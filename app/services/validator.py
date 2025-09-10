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
from app.services.url_parser import URLParser
from app.services.instagram_scraper import InstagramScraper
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
        self.url_parser = URLParser()
        self.scraper = InstagramScraper()
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

            # Step 1: Parse URL and extract username
            url_username, url_parsing_success = await self._parse_url(
                submission.instagram_url, validation_result
            )
            submission.url_username = url_username

            # Step 2: Check account visibility (if we have username)
            is_public = None
            account_check_success = False
            if url_username:
                is_public, account_check_success = await self._check_account_visibility(
                    url_username, validation_result
                )
                submission.is_account_public = is_public

            # Step 3: Extract content data
            extracted_data, extraction_success = await self._extract_content_data(
                submission, is_public, validation_result
            )

            if extracted_data:
                submission.content_username = extracted_data.username
                submission.extracted_hashtags = extracted_data.hashtags
                submission.extracted_data = {
                    "content_type": extracted_data.content_type.value if extracted_data.content_type else None,
                    "confidence": extracted_data.confidence,
                    "extraction_method": extracted_data.extraction_method
                }

            # Step 4: Validate username match
            username_match, username_validation_success = self._validate_username_match(
                url_username, extracted_data.username if extracted_data else None, validation_result
            )
            submission.username_match = username_match

            # Step 5: Validate hashtags
            hashtags_valid, missing_hashtags, hashtag_validation_success = self._validate_hashtags(
                extracted_data.hashtags if extracted_data else [], validation_result
            )
            submission.hashtags_valid = hashtags_valid
            submission.missing_hashtags = missing_hashtags

            # Determine overall validation status
            overall_success = all([
                url_parsing_success or extraction_success,  # Either URL parsing or extraction must work
                extraction_success,
                username_validation_success,
                hashtag_validation_success
            ])

            if overall_success and username_match and hashtags_valid:
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
                "username_match": username_match,
                "hashtags_valid": hashtags_valid,
                "missing_hashtags": missing_hashtags,
                "is_account_public": is_public,
                "extraction_method": extracted_data.extraction_method if extracted_data else None,
                "confidence": extracted_data.confidence if extracted_data else None
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

    async def _parse_url(self, url: str, validation_result: ValidationResult) -> Tuple[Optional[str], bool]:
        """Parse URL and extract username."""
        try:
            # Validate URL format
            is_valid, error_msg = self.url_parser.validate_instagram_url(url)
            if not is_valid:
                raise ValidationError(error_msg)

            # Extract username
            username = self.url_parser.extract_username_from_url(url)

            validation_result.url_parsing_success = True
            return username, True

        except Exception as e:
            logger.warning(f"URL parsing failed: {str(e)}")
            validation_result.url_parsing_success = False
            return None, False

    async def _check_account_visibility(self, username: str, validation_result: ValidationResult) -> Tuple[
        Optional[bool], bool]:
        """Check if account is public or private."""
        try:
            is_public = await self.scraper.check_account_visibility(username)
            validation_result.account_check_success = True
            return is_public, True

        except Exception as e:
            logger.warning(f"Account visibility check failed: {str(e)}")
            validation_result.account_check_success = False
            return None, False

    async def _extract_content_data(self, submission: Submission, is_public: Optional[bool],
                                    validation_result: ValidationResult) -> Tuple[Optional[ExtractedData], bool]:
        """Extract content data using scraping or LLM analysis."""
        try:
            extracted_data = None

            # Try scraping first for public accounts
            if is_public:
                try:
                    extracted_data = await self.scraper.scrape_post_data(submission.instagram_url)
                    validation_result.extraction_method = "scraping"
                    validation_result.raw_scraped_data = {
                        "username": extracted_data.username,
                        "hashtags": extracted_data.hashtags,
                        "content_type": extracted_data.content_type.value if extracted_data.content_type else None
                    }
                    logger.info("Successfully extracted data via scraping")
                except Exception as e:
                    logger.warning(f"Scraping failed, falling back to LLM analysis: {str(e)}")

            # Fall back to LLM analysis if scraping failed or account is private
            if not extracted_data:
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
            logger.error(f"Content extraction failed: {str(e)}")
            validation_result.content_extraction_success = False
            return None, False

    def _validate_username_match(self, url_username: Optional[str], content_username: Optional[str],
                                 validation_result: ValidationResult) -> Tuple[Optional[bool], bool]:
        """Validate that usernames match."""
        try:
            if not url_username and not content_username:
                validation_result.username_validation_success = False
                return None, False

            if not url_username or not content_username:
                # If we only have one username, we can't validate match
                # but this might still be acceptable depending on extraction method
                validation_result.username_validation_success = True
                return None, True

            # Normalize usernames for comparison
            url_username_clean = url_username.lower().strip()
            content_username_clean = content_username.lower().strip()

            # Remove @ symbol if present
            if url_username_clean.startswith('@'):
                url_username_clean = url_username_clean[1:]
            if content_username_clean.startswith('@'):
                content_username_clean = content_username_clean[1:]

            username_match = url_username_clean == content_username_clean
            validation_result.username_validation_success = True

            return username_match, True

        except Exception as e:
            logger.error(f"Username validation failed: {str(e)}")
            validation_result.username_validation_success = False
            return False, False

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
            from sqlmodel import select, func

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

            return {
                "total_submissions": total_count,
                "valid_submissions": valid_count,
                "invalid_submissions": invalid_count,
                "pending_submissions": pending_count,
                "error_submissions": error_count,
                "success_rate": round(success_rate, 2)
            }

        finally:
            session.close()