"""
Service for analyzing Instagram screenshots using OpenAI GPT-4 Vision.
"""

import json
import base64
from typing import Optional, Dict, Any

import aiofiles
import openai
from openai import AsyncOpenAI
from app.config import settings
from app.utils.exceptions import OpenAIAnalysisError
from app.models.api import ExtractedData
from app.models.database import ContentType


class OpenAIAnalyzer:
    """Service for analyzing Instagram content using OpenAI."""

    def __init__(self):
        """Initialize OpenAI client."""
        if not settings.openai_api_key:
            raise OpenAIAnalysisError("OpenAI API key not configured")

        self.client = AsyncOpenAI(api_key=settings.openai_api_key , base_url="https://api.avalapis.ir/v1")

    async def analyze_screenshot(self, image_path: str, expected_content_type: ContentType) -> ExtractedData:
        """
        Analyze Instagram screenshot using GPT-4 Vision.

        Args:
            image_path: Path to the screenshot image
            expected_content_type: Expected content type from user input

        Returns:
            ExtractedData: Extracted data from the screenshot

        Raises:
            OpenAIAnalysisError: If analysis fails
        """
        try:
            # Encode image to base64
            base64_image = await self._encode_image_to_base64(image_path)

            # Create the prompt for analysis
            prompt = self._create_analysis_prompt(expected_content_type)

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent results
            )

            # Parse response
            content = response.choices[0].message.content
            return self._parse_analysis_response(content, expected_content_type)

        except Exception as e:
            raise OpenAIAnalysisError(f"Failed to analyze screenshot: {str(e)}")

    async def _encode_image_to_base64(self, image_path: str, max_width: int = 800) -> str:
        """Encode and resize image to reduce token usage."""
        try:
            from PIL import Image
            import io

            # Open and resize image
            with Image.open(image_path) as img:
                # Calculate new dimensions maintaining aspect ratio
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.LANCZOS)

                # Convert to JPEG with optimized quality
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                image_data = buffer.getvalue()

            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            # Fallback to original method
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
                return base64.b64encode(image_data).decode('utf-8')

    def _create_analysis_prompt(self, expected_content_type: ContentType) -> str:
        """Create analysis prompt for OpenAI."""
        return f"""
        You are analyzing an Instagram screenshot. extract the following information and return it as a JSON object:

        1. **username**: The Instagram username (without @ symbol)
        2. **hashtags**: All hashtags visible in the content (as an array of strings, including the # symbol)
        3. **content_type**: The type of content ("post", "story", or "reels")
        4. **confidence**: Your confidence level in the extraction (0.0 to 1.0)

        Expected content type from user: {expected_content_type.value}

        **Important Instructions:**
        - Look for the username in the header area of the Instagram interface it's usually at the top of the post
        - Extract ALL visible hashtags from the caption or text and should include the # symbol (e.g., "#hashtag")
        - If you cannot clearly see certain information, set confidence lower
        - Return ONLY a valid JSON object, no additional text

        **JSON Format:**
        {{
            "username": "extracted_username",
            "hashtags": ["#hashtag1", "#hashtag2"],
            "content_type": "post|story|reels",
            "confidence": 0.95
        }}
        """

    def _parse_analysis_response(self, response_content: str, expected_content_type: ContentType) -> ExtractedData:
        """Parse OpenAI response and create ExtractedData object."""
        try:
            # Clean response content (remove any markdown formatting)
            clean_content = response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()

            # Parse JSON
            data = json.loads(clean_content)

            # Validate required fields
            if "username" not in data or "hashtags" not in data:
                raise OpenAIAnalysisError("Missing required fields in OpenAI response")

            # Clean and validate username
            username = data["username"].strip().lower()
            if username.startswith('@'):
                username = username[1:]

            # Clean and validate hashtags
            hashtags = []
            if isinstance(data["hashtags"], list):
                for tag in data["hashtags"]:
                    if isinstance(tag, str):
                        clean_tag = tag.strip().lower()
                        if not clean_tag.startswith('#'):
                            clean_tag = f"#{clean_tag}"
                        hashtags.append(clean_tag)

            # Validate content type
            content_type_str = data.get("content_type", expected_content_type.value).lower()
            try:
                content_type = ContentType(content_type_str)
            except ValueError:
                content_type = expected_content_type

            # Get confidence
            confidence = float(data.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1

            return ExtractedData(
                username=username,
                hashtags=hashtags,
                content_type=content_type,
                confidence=confidence,
                extraction_method="llm"
            )

        except json.JSONDecodeError as e:
            raise OpenAIAnalysisError(f"Failed to parse OpenAI JSON response: {str(e)}")
        except Exception as e:
            raise OpenAIAnalysisError(f"Failed to process OpenAI response: {str(e)}")

    async def validate_extraction_quality(self, extracted_data: ExtractedData) -> Dict[str, Any]:
        """
        Validate the quality of extracted data.

        Args:
            extracted_data: Data extracted from analysis

        Returns:
            Dict[str, Any]: Quality assessment
        """
        quality_issues = []

        # Check username validity
        if not extracted_data.username or len(extracted_data.username) < 1:
            quality_issues.append("Username is empty or too short")

        # Check if username contains invalid characters
        if extracted_data.username:
            import re
            if not re.match(r'^[a-zA-Z0-9_.]+$', extracted_data.username):
                quality_issues.append("Username contains invalid characters")

        # Check hashtags
        if not extracted_data.hashtags:
            quality_issues.append("No hashtags found")
        else:
            for hashtag in extracted_data.hashtags:
                if not hashtag.startswith('#'):
                    quality_issues.append(f"Invalid hashtag format: {hashtag}")

        # Assess overall quality
        quality_score = extracted_data.confidence or 0.0
        if quality_issues:
            quality_score *= 0.7  # Reduce score if there are issues

        return {
            "quality_score": quality_score,
            "quality_issues": quality_issues,
            "is_reliable": quality_score >= 0.7 and len(quality_issues) == 0
        }