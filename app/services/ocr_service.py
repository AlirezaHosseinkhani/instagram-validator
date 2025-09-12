"""
Service for extracting text from images using Tesseract OCR with enhanced Persian support.
"""

import asyncio
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import re
from typing import List, Optional, Dict, Any, Tuple
import logging
from app.models.api import ExtractedData
from app.models.database import ContentType
from app.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class OCRService:
    """Enhanced OCR service with improved Persian text extraction."""

    def __init__(self):
        """Initialize OCR service with Persian optimization."""
        # Configure tesseract if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux

        # Enhanced Persian character sets
        self.persian_chars = 'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ'
        self.persian_digits = '۰۱۲۳۴۵۶۷۸۹'
        self.arabic_chars = 'ءأإؤئابتثجحخدذرزسشصضطظعغفقكلمنهوي'

        # Combined character set for better recognition
        self.all_persian_arabic = self.persian_chars + self.arabic_chars + self.persian_digits

        # Zero-width characters that can interfere
        self.zwj = '\u200D'  # Zero Width Joiner
        self.zwnj = '\u200C'  # Zero Width Non-Joiner

    async def extract_text_from_image(self, image_path: str, language: str = 'eng+fas') -> str:
        """
        Extract text from image using multiple OCR strategies for Persian support.

        Args:
            image_path: Path to the image file
            language: Language codes for OCR

        Returns:
            str: Extracted text
        """
        try:
            # Try multiple extraction strategies
            strategies = [
                ('preprocessed_persian', 'fas+eng'),
                ('standard_persian', 'eng+fas'),
                ('persian_only', 'fas'),
                ('fallback_english', 'eng')
            ]

            best_result = ""
            best_score = 0

            for strategy_name, lang in strategies:
                try:
                    if strategy_name == 'preprocessed_persian':
                        result = await self._extract_with_preprocessing(image_path, lang)
                    else:
                        result = await self._extract_standard(image_path, lang)

                    if result:
                        # Score the result based on Persian content and length
                        score = self._score_extraction_result(result)
                        logger.info(f"Strategy '{strategy_name}' score: {score:.2f}")

                        if score > best_score:
                            best_result = result
                            best_score = score

                except Exception as e:
                    logger.warning(f"Strategy '{strategy_name}' failed: {str(e)}")
                    continue

            logger.info(f"Best extraction score: {best_score:.2f}, text length: {len(best_result)}")
            return best_result

        except Exception as e:
            logger.warning(f"All OCR strategies failed: {str(e)}")
            return ""

    async def _extract_with_preprocessing(self, image_path: str, language: str) -> str:
        """Extract text with advanced image preprocessing for Persian."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._extract_preprocessed_sync,
            image_path,
            language
        )

    async def _extract_standard(self, image_path: str, language: str) -> str:
        """Standard text extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._extract_text_sync,
            image_path,
            language
        )

    def _extract_preprocessed_sync(self, image_path: str, language: str) -> str:
        """Synchronous extraction with Persian-optimized preprocessing."""
        try:
            # Load image with OpenCV for advanced preprocessing
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Apply Persian-optimized preprocessing
            processed_img = self._preprocess_for_persian(img)

            # Convert back to PIL for tesseract
            pil_img = Image.fromarray(processed_img)

            # Persian-optimized tesseract config
            custom_config = self._get_persian_config(language)

            # Extract text
            text = pytesseract.image_to_string(pil_img, config=custom_config)

            # Post-process Persian text
            cleaned_text = self._clean_persian_text(text)

            logger.info(f"Preprocessed Persian OCR: {len(cleaned_text)} chars")
            return cleaned_text

        except Exception as e:
            logger.warning(f"Preprocessed Persian OCR failed: {str(e)}")
            return ""

    def _preprocess_for_persian(self, img: np.ndarray) -> np.ndarray:
        """Apply Persian-specific image preprocessing."""
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            # Resize if too small (Persian text needs good resolution)
            height, width = gray.shape
            if height < 100 or width < 100:
                scale = max(2.0, 100 / min(height, width))
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)

            # Enhance contrast using CLAHE (good for Persian text)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Morphological operations to connect Persian characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

            # Sharpening filter
            kernel_sharpen = np.array([[-1, -1, -1],
                                     [-1,  9, -1],
                                     [-1, -1, -1]])
            sharpened = cv2.filter2D(morphed, -1, kernel_sharpen)

            # Ensure good contrast
            sharpened = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=10)

            return sharpened

        except Exception as e:
            logger.warning(f"Persian preprocessing failed: {str(e)}")
            return img

    def _get_persian_config(self, language: str) -> str:
        """Get optimized tesseract config for Persian text."""
        # Include Persian-specific characters and common symbols
        allowed_chars = (
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            + self.all_persian_arabic + '#@_.,!?()[]{}"\'-+='
            + self.zwj + self.zwnj  # Include zero-width joiners
        )

        # Persian-optimized OCR settings
        config = (
            f'--oem 3 --psm 6 -l {language} '
            f'-c tessedit_char_whitelist={allowed_chars} '
            f'-c preserve_interword_spaces=1 '
            f'-c textord_really_old_xheight=1 '
            f'-c textord_min_xheight=14 '
            f'-c load_system_dawg=0 '
            f'-c load_freq_dawg=0'
        )

        return config

    def _extract_text_sync(self, image_path: str, language: str = 'eng+fas') -> str:
        """Standard synchronous text extraction with Persian support."""
        try:
            image = Image.open(image_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Basic enhancement for Persian text
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)

            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)

            # Standard Persian config
            custom_config = self._get_persian_config(language)

            text = pytesseract.image_to_string(image, config=custom_config)
            cleaned_text = self._clean_persian_text(text)

            return cleaned_text

        except Exception as e:
            logger.warning(f"Standard Persian OCR failed: {str(e)}")
            return ""

    def _clean_persian_text(self, text: str) -> str:
        """Clean and normalize Persian text."""
        if not text:
            return ""

        try:
            # Normalize Persian characters
            text = self._normalize_persian_chars(text)

            # Fix common OCR mistakes for Persian
            text = self._fix_persian_ocr_mistakes(text)

            # Clean up whitespace but preserve Persian text structure
            lines = text.split('\n')
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                if line:  # Keep non-empty lines
                    cleaned_lines.append(line)

            return '\n'.join(cleaned_lines)

        except Exception as e:
            logger.warning(f"Persian text cleaning failed: {str(e)}")
            return text

    def _normalize_persian_chars(self, text: str) -> str:
        """Normalize Persian characters to standard forms."""
        # Common Persian character normalizations
        normalizations = {
            'ك': 'ک',  # Arabic kaf to Persian kaf
            'ي': 'ی',  # Arabic yeh to Persian yeh
            'ء': '',   # Remove hamza if standalone
            'أ': 'ا',  # Alef with hamza above to alef
            'إ': 'ا',  # Alef with hamza below to alef
            'ؤ': 'و',  # Waw with hamza to waw
            'ئ': 'ی',  # Yeh with hamza to yeh
        }

        for old_char, new_char in normalizations.items():
            text = text.replace(old_char, new_char)

        return text

    def _fix_persian_ocr_mistakes(self, text: str) -> str:
        """Fix common OCR mistakes in Persian text."""
        # Common OCR mistakes in Persian
        fixes = {
            'رن': 'ری',  # Common mistake
            'لا': 'ال',  # Lam-alef issues
            'تر': 'ته',  # Te-he confusion
            ' ه ': 'ه',   # Isolated heh
            'ن ': 'ن',    # Nun spacing issues
        }

        for mistake, correction in fixes.items():
            text = text.replace(mistake, correction)

        return text

    def _score_extraction_result(self, text: str) -> float:
        """Score extraction result based on Persian content quality."""
        if not text:
            return 0.0

        score = 0.0

        # Base score for having text
        score += min(0.3, len(text) / 100)

        # Bonus for Persian characters
        persian_chars = sum(1 for c in text if c in self.all_persian_arabic)
        if persian_chars > 0:
            score += min(0.4, persian_chars / 20)

        # Bonus for hashtags (both Persian and English)
        hashtag_count = len(re.findall(r'#[\w\u0600-\u06FF\u200C\u200D_]+', text))
        score += min(0.3, hashtag_count * 0.1)

        # Penalty for too much noise (random characters)
        noise_ratio = sum(1 for c in text if not (c.isalnum() or c in self.all_persian_arabic or c.isspace() or c in '#@_.,!?()[]{}"\'-+=')) / max(1, len(text))
        score -= noise_ratio * 0.2

        return max(0.0, min(1.0, score))

    async def extract_hashtags_from_image(self, image_path: str) -> List[str]:
        """Extract hashtags with enhanced Persian support."""
        try:
            # Get text using the enhanced extraction
            text = await self.extract_text_from_image(image_path, 'fas+eng')

            if not text:
                logger.info("No text extracted for hashtag detection")
                return []

            # Extract hashtags with Persian support
            hashtags = self._extract_hashtags_from_text(text)

            # Try additional extraction with different preprocessing if few hashtags found
            if len(hashtags) < 2:
                additional_text = await self._extract_with_preprocessing(image_path, 'fas')
                additional_hashtags = self._extract_hashtags_from_text(additional_text)

                # Merge results
                all_hashtags = hashtags + additional_hashtags
                hashtags = self._deduplicate_hashtags(all_hashtags)

            logger.info(f"OCR extracted {len(hashtags)} hashtags: {hashtags}")
            return hashtags

        except Exception as e:
            logger.warning(f"Persian hashtag extraction failed: {str(e)}")
            return []

    def _extract_hashtags_from_text(self, text: str) -> List[str]:
        """Enhanced hashtag extraction with Persian support."""
        try:
            # Multiple regex patterns for better coverage
            patterns = [
                r'#[\w\u0600-\u06FF\u200C\u200D_]+',  # Standard pattern
                r'＃[\w\u0600-\u06FF\u200C\u200D_]+',  # Full-width hash
                r'[#＃][\u0600-\u06FF]+[\w\u0600-\u06FF\u200C\u200D_]*',  # Persian-first
            ]

            all_hashtags = []

            for pattern in patterns:
                matches = re.findall(pattern, text)
                all_hashtags.extend(matches)

            # Clean and validate hashtags
            cleaned_hashtags = []
            for hashtag in all_hashtags:
                clean_hashtag = self._clean_hashtag(hashtag)
                if clean_hashtag:
                    cleaned_hashtags.append(clean_hashtag)

            return self._deduplicate_hashtags(cleaned_hashtags)

        except Exception as e:
            logger.warning(f"Hashtag pattern matching failed: {str(e)}")
            return []

    def _clean_hashtag(self, hashtag: str) -> Optional[str]:
        """Enhanced hashtag cleaning with Persian support."""
        try:
            if not hashtag:
                return None

            # Normalize hash symbol
            if hashtag.startswith('＃'):
                hashtag = '#' + hashtag[1:]

            if not hashtag.startswith('#'):
                return None

            # Remove hash and clean
            clean_tag = hashtag[1:].strip()

            # Remove problematic zero-width characters at edges
            clean_tag = clean_tag.strip(self.zwj + self.zwnj)

            # Normalize Persian characters
            clean_tag = self._normalize_persian_chars(clean_tag)

            # Length validation
            if len(clean_tag) < 2 or len(clean_tag) > 50:
                return None

            # Character validation (more permissive for Persian)
            valid_chars = set(
                'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
                + self.all_persian_arabic + self.zwj + self.zwnj
            )

            # Filter invalid characters but keep the hashtag if it has valid content
            filtered_tag = ''.join(c for c in clean_tag if c in valid_chars)

            if len(filtered_tag) < 2:
                return None

            # Must contain at least one letter (Persian or English)
            has_letter = any(c.isalpha() or c in self.all_persian_arabic for c in filtered_tag)
            if not has_letter:
                return None

            return f"#{filtered_tag}"

        except Exception as e:
            logger.warning(f"Hashtag cleaning failed for '{hashtag}': {str(e)}")
            return None

    def _deduplicate_hashtags(self, hashtags: List[str]) -> List[str]:
        """Remove duplicate hashtags while preserving order."""
        seen = set()
        unique_hashtags = []

        for hashtag in hashtags:
            # Normalize for comparison (case-insensitive, remove zero-width chars)
            normalized = hashtag.lower().replace(self.zwj, '').replace(self.zwnj, '')

            if normalized not in seen:
                unique_hashtags.append(hashtag)
                seen.add(normalized)

        return unique_hashtags

    def _contains_persian(self, text: str) -> bool:
        """Enhanced Persian character detection."""
        try:
            # Check for Persian/Arabic Unicode ranges
            persian_pattern = r'[\u0600-\u06FF]'
            return bool(re.search(persian_pattern, text))
        except Exception:
            return False

    async def analyze_screenshot_with_ocr(self, image_path: str, content_type: ContentType) -> Optional[ExtractedData]:
        """Enhanced screenshot analysis with Persian support."""
        try:
            # Extract hashtags and username in parallel
            hashtags_task = self.extract_hashtags_from_image(image_path)
            username_task = self.extract_username_from_image(image_path)

            hashtags, username = await asyncio.gather(hashtags_task, username_task)

            # Enhanced logging
            logger.info(f"OCR Results - Username: {username}, Hashtags: {len(hashtags)} found")
            if hashtags:
                persian_hashtags = [h for h in hashtags if self._contains_persian(h)]
                english_hashtags = [h for h in hashtags if not self._contains_persian(h)]
                logger.info(f"Persian hashtags: {persian_hashtags}")
                logger.info(f"English hashtags: {english_hashtags}")

            # Return data if we found something useful
            if hashtags or username:
                # Enhanced confidence calculation
                confidence = self._calculate_ocr_confidence(hashtags, username)

                return ExtractedData(
                    username=username,
                    hashtags=hashtags,
                    content_type=content_type,
                    confidence=confidence,
                    extraction_method="ocr_enhanced_persian"
                )

            logger.info("Enhanced OCR found no useful data")
            return None

        except Exception as e:
            logger.warning(f"Enhanced OCR analysis failed: {str(e)}")
            return None

    def _calculate_ocr_confidence(self, hashtags: List[str], username: Optional[str]) -> float:
        """Calculate confidence score for OCR extraction."""
        confidence = 0.0

        if hashtags:
            confidence += 0.4  # Base confidence for hashtags
            confidence += min(0.25, len(hashtags) * 0.05)  # Bonus for multiple hashtags

            # Higher confidence for Persian hashtags (harder to extract correctly)
            persian_count = sum(1 for h in hashtags if self._contains_persian(h))
            if persian_count > 0:
                confidence += min(0.15, persian_count * 0.05)
                logger.info(f"Persian hashtag bonus: {persian_count} hashtags")

        if username:
            confidence += 0.25  # Bonus for username

        # Cap OCR confidence (it's less reliable than API methods)
        return min(0.8, confidence)

    # Keep existing methods for username extraction and other functionality
    async def extract_username_from_image(self, image_path: str) -> Optional[str]:
        """Extract Instagram username from image using OCR."""
        try:
            # Try both multilingual and English-only extraction
            text_multilang = await self.extract_text_from_image(image_path, 'eng+fas')
            text_eng = await self.extract_text_from_image(image_path, 'eng')

            combined_text = f"{text_multilang}\n{text_eng}"

            if not combined_text.strip():
                return None

            username = self._extract_username_from_text(combined_text)

            if username:
                logger.info(f"OCR extracted username: {username}")

            return username

        except Exception as e:
            logger.warning(f"Username extraction from OCR failed: {str(e)}")
            return None

    def _extract_username_from_text(self, text: str) -> Optional[str]:
        """Extract username from text using patterns."""
        try:
            patterns = [
                r'@([a-zA-Z0-9_.]+)',
                r'instagram\.com/([a-zA-Z0-9_.]+)',
                r'(?:^|\s)([a-zA-Z0-9_.]{3,30})(?:\s|$)',
            ]

            lines = text.split('\n')

            for pattern in patterns:
                for line in lines:
                    matches = re.findall(pattern, line.strip())
                    for match in matches:
                        if self._is_valid_instagram_username(match):
                            return match.lower()

            return None

        except Exception as e:
            logger.warning(f"Username pattern matching failed: {str(e)}")
            return None

    def _is_valid_instagram_username(self, username: str) -> bool:
        """Validate Instagram username format."""
        try:
            if not username or len(username) > 30 or len(username) < 1:
                return False

            if username.startswith('.') or username.endswith('.'):
                return False

            if '..' in username:
                return False

            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._')
            if not all(c in allowed_chars for c in username):
                return False

            if not any(c.isalnum() for c in username):
                return False

            common_mistakes = ['www', 'http', 'https', 'com', 'instagram', 'post', 'reel']
            if username.lower() in common_mistakes:
                return False

            return True

        except Exception:
            return False