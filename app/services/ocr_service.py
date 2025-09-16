"""
Enhanced OCR service for extracting Instagram usernames and hashtags from images.
Optimized for both Persian and English text with improved accuracy.
"""

import asyncio
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import re
from typing import List, Optional, Tuple, Dict, Any
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time

from app.models.api import ExtractedData
from app.models.database import ContentType
from app.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class OCRService:
    """Enhanced OCR service optimized for both English and Persian hashtag extraction."""

    def __init__(self, max_workers: int = 3):
        """Initialize OCR service with multi-language optimizations."""
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Character sets for validation
        self.persian_chars = 'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ'
        self.persian_digits = '۰۱۲۳۴۵۶۷۸۹'
        self.arabic_chars = 'ءأإؤئابتثجحخدذرزسشصضطظعغفقكلمنهوي'
        self.all_persian_arabic = self.persian_chars + self.arabic_chars + self.persian_digits

        # Zero-width characters
        self.zwj = '\u200D'
        self.zwnj = '\u200C'

        # Precompiled regex patterns for performance
        self._compile_patterns()

        # Character normalization mapping
        self.char_normalizations = {
            'ك': 'ک', 'ي': 'ی', 'ء': '', 'أ': 'ا',
            'إ': 'ا', 'ؤ': 'و', 'ئ': 'ی'
        }

        # OCR common mistakes for Persian numbers/letters
        self.persian_ocr_fixes = {
            '3': '۳', '0': '۰', '1': '۱', '2': '۲', '4': '۴',
            '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹',
            'mazdad': 'mazda3', 'mix80': 'mx30', 'bmwx3': 'bmwx3',
            'mercedesamg': 'mercedesamg', 'audiA4': 'audia4'
        }

    def _compile_patterns(self) -> None:
        """Precompile regex patterns for better performance."""
        # Username patterns (prioritized by reliability)
        self.username_patterns = [
            re.compile(r'@([a-zA-Z0-9][a-zA-Z0-9_.]{0,28}[a-zA-Z0-9])'),
            re.compile(r'instagram\.com/([a-zA-Z0-9][a-zA-Z0-9_.]{0,28}[a-zA-Z0-9])'),
            re.compile(r'(?:^|\s)([a-zA-Z0-9][a-zA-Z0-9_.]{2,28}[a-zA-Z0-9])(?=\s|$)'),
        ]

        # Enhanced hashtag patterns for both languages
        self.hashtag_patterns = [
            re.compile(r'#([\w\u0600-\u06FF\u200C\u200D_]+)'),  # Standard pattern
            re.compile(r'＃([\w\u0600-\u06FF\u200C\u200D_]+)'),  # Full-width hash
            re.compile(r'[#＃]([\u0600-\u06FF]+[\w\u0600-\u06FF\u200C\u200D_]*)'),  # Persian-first
            re.compile(r'[#＃]([a-zA-Z]+[\w\u0600-\u06FF\u200C\u200D_]*)')  # English-first
        ]

        # Persian character detection
        self.persian_pattern = re.compile(r'[\u0600-\u06FF]')
        self.english_pattern = re.compile(r'[a-zA-Z]')

    @lru_cache(maxsize=15)
    def _get_tesseract_config(self, language: str, mode: str = 'standard') -> str:
        """Get cached tesseract configuration optimized for different modes."""
        if mode == 'username':
            # Optimized for username extraction (English only)
            return f'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._@'
        elif mode == 'persian_hashtags':
            # Optimized for Persian hashtags with enhanced settings
            return (f'--oem 3 --psm 6 -l {language} '
                   f'-c preserve_interword_spaces=1 '
                   f'-c textord_really_old_xheight=1 '
                   f'-c textord_min_xheight=14 '
                   f'-c textord_force_make_prop_words=1 '
                   f'-c chop_enable=1 '
                   f'-c use_new_state_cost=1 '
                   f'-c language_model_ngram_on=0')
        elif mode == 'english_hashtags':
            # Optimized for English hashtags and mixed content
            return f'--oem 3 --psm 6 -l {language} -c preserve_interword_spaces=1'

        return f'--oem 3 --psm 6 -l {language}'

    def _preprocess_image_for_language(self, image_path: str, target_lang: str) -> np.ndarray:
        """Language-specific image preprocessing."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

            # Resize if too small
            height, width = gray.shape
            if height < 200 or width < 200:
                scale = max(1.5, 200 / min(height, width))
                new_size = (int(width * scale), int(height * scale))
                gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

            if target_lang == 'persian':
                # Persian-optimized preprocessing (from your original code)
                # Noise reduction
                denoised = cv2.fastNlMeansDenoising(gray, h=10)

                # Enhance contrast using CLAHE (good for Persian text)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)

                # Morphological operations to connect Persian characters
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

                # Sharpening filter
                kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(morphed, -1, kernel_sharpen)

                # Ensure good contrast
                result = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=10)

            elif target_lang == 'english':
                # English-optimized preprocessing
                # Noise reduction with different parameters
                denoised = cv2.fastNlMeansDenoising(gray, h=8)

                # Adaptive thresholding for clear English text
                adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)

                # Light morphological operation
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                result = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

            else:  # mixed
                # Balanced preprocessing for mixed content
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Light denoising
                denoised = cv2.fastNlMeansDenoising(enhanced, h=6)

                # Moderate sharpening
                kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                result = cv2.filter2D(denoised, -1, kernel_sharpen)

            return result

        except Exception as e:
            logger.warning(f"Image preprocessing failed for {target_lang}: {e}")
            # Fallback to basic preprocessing
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.array([])

    async def _extract_text_with_strategy(self, image_path: str, language: str,
                                        target_lang: str, mode: str) -> str:
        """Extract text using specific language strategy."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_text_sync_strategy,
            image_path,
            language,
            target_lang,
            mode
        )

    def _extract_text_sync_strategy(self, image_path: str, language: str,
                                  target_lang: str, mode: str) -> str:
        """Synchronous text extraction with specific strategy."""
        try:
            # Use language-specific preprocessing
            processed_img = self._preprocess_image_for_language(image_path, target_lang)
            if processed_img.size == 0:
                return ""

            # Convert to PIL Image
            pil_img = Image.fromarray(processed_img)

            # Get appropriate config
            config = self._get_tesseract_config(language, mode)

            # Extract text
            text = pytesseract.image_to_string(pil_img, config=config)

            # Apply language-specific fixes
            # if target_lang == 'persian':
            #     text = self._fix_persian_ocr_mistakes(text)

            return self._normalize_text(text)

        except Exception as e:
            logger.warning(f"Text extraction failed for {target_lang}: {e}")
            return ""

    def _fix_persian_ocr_mistakes(self, text: str) -> str:
        """Fix common OCR mistakes in Persian text and mixed content."""
        if not text:
            return ""

        # Apply common fixes
        for mistake, correction in self.persian_ocr_fixes.items():
            text = text.replace(mistake, correction)

        # Fix common Persian OCR issues
        persian_fixes = {
            'رن': 'ری', 'لا': 'ال', 'تر': 'ته', ' ه ': 'ه', 'ن ': 'ن'
        }

        for mistake, correction in persian_fixes.items():
            text = text.replace(mistake, correction)

        return text

    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization for both languages."""
        if not text:
            return ""

        # Normalize Persian characters
        for old_char, new_char in self.char_normalizations.items():
            text = text.replace(old_char, new_char)

        # Clean whitespace while preserving structure
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

    async def extract_hashtags_from_image(self, image_path: str) -> List[str]:
        """Enhanced hashtag extraction using multiple strategies for both languages."""
        try:
            start_time = time.time()

            # Strategy 1: Persian-optimized extraction (your original approach)
            persian_task = self._extract_text_with_strategy(
                image_path, 'fas+eng', 'persian', 'persian_hashtags'
            )

            # Strategy 2: English-optimized extraction
            english_task = self._extract_text_with_strategy(
                image_path, 'eng+fas', 'english', 'english_hashtags'
            )

            # Strategy 3: Mixed content extraction
            mixed_task = self._extract_text_with_strategy(
                image_path, 'fas+eng', 'mixed', 'standard'
            )

            # Execute all strategies concurrently
            persian_text, english_text, mixed_text = await asyncio.gather(
                persian_task, english_task, mixed_task
            )

            # Extract hashtags from each result
            all_hashtags = []

            # Process Persian-optimized text
            if persian_text:
                persian_hashtags = self._extract_hashtags_from_text(persian_text, 'persian')
                all_hashtags.extend(persian_hashtags)
                logger.info(f"Persian strategy found {len(persian_hashtags)} hashtags")

            # Process English-optimized text
            if english_text:
                english_hashtags = self._extract_hashtags_from_text(english_text, 'english')
                all_hashtags.extend(english_hashtags)
                logger.info(f"English strategy found {len(english_hashtags)} hashtags")

            # Process mixed content text
            if mixed_text:
                mixed_hashtags = self._extract_hashtags_from_text(mixed_text, 'mixed')
                all_hashtags.extend(mixed_hashtags)
                logger.info(f"Mixed strategy found {len(mixed_hashtags)} hashtags")

            # Merge and deduplicate results intelligently
            final_hashtags = self._merge_hashtag_results(all_hashtags)

            extraction_time = time.time() - start_time
            logger.info(f"Total hashtag extraction took {extraction_time:.2f}s, found {len(final_hashtags)} unique hashtags: {final_hashtags}")

            return final_hashtags

        except Exception as e:
            logger.error(f"Hashtag extraction failed: {e}")
            return []

    def _extract_hashtags_from_text(self, text: str, strategy: str = 'mixed') -> List[str]:
        """Extract hashtags with strategy-specific processing."""
        if not text:
            return []

        try:
            hashtags = []

            # Use all patterns but prioritize based on strategy
            for pattern in self.hashtag_patterns:
                matches = pattern.findall(text)
                hashtags.extend(matches)

            # Clean and validate hashtags
            cleaned_hashtags = []
            for hashtag in hashtags:
                clean_hashtag = self._clean_hashtag(hashtag, strategy)
                if clean_hashtag:
                    cleaned_hashtags.append(clean_hashtag)

            return cleaned_hashtags

        except Exception as e:
            logger.warning(f"Hashtag extraction failed for {strategy}: {e}")
            return []

    def _clean_hashtag(self, hashtag: str, strategy: str = 'mixed') -> Optional[str]:
        """Clean and validate hashtag with strategy-specific rules."""
        if not hashtag:
            return None

        try:
            # Remove problematic characters at edges
            clean_tag = hashtag.strip(self.zwj + self.zwnj + ' \t\n\r')

            # Normalize Persian characters
            for old_char, new_char in self.char_normalizations.items():
                clean_tag = clean_tag.replace(old_char, new_char)

            # Strategy-specific cleaning
            if strategy == 'persian':
                # More lenient for Persian hashtags
                min_length = 1
                max_length = 60
            elif strategy == 'english':
                # Stricter for English hashtags
                min_length = 2
                max_length = 40
            else:
                # Balanced for mixed content
                min_length = 2
                max_length = 50

            # Length validation
            if len(clean_tag) < min_length or len(clean_tag) > max_length:
                return None

            # Character validation
            valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') | set(self.all_persian_arabic) | {self.zwj, self.zwnj}
            filtered_tag = ''.join(c for c in clean_tag if c in valid_chars)

            if len(filtered_tag) < min_length:
                return None

            # Must contain at least one letter
            if not any(c.isalpha() or c in self.all_persian_arabic for c in filtered_tag):
                return None

            return f"#{filtered_tag}"

        except Exception as e:
            logger.warning(f"Hashtag cleaning failed for '{hashtag}': {e}")
            return None

    def _merge_hashtag_results(self, all_hashtags: List[str]) -> List[str]:
        """Intelligently merge hashtag results from different strategies."""
        if not all_hashtags:
            return []

        # Group hashtags by similarity
        hashtag_groups = {}

        for hashtag in all_hashtags:
            if not hashtag:
                continue

            # Normalize for grouping (remove # and convert to lowercase)
            normalized = hashtag[1:].lower().replace(self.zwj, '').replace(self.zwnj, '')

            if normalized not in hashtag_groups:
                hashtag_groups[normalized] = []
            hashtag_groups[normalized].append(hashtag)

        # Select the best hashtag from each group
        final_hashtags = []

        for normalized, group in hashtag_groups.items():
            if len(group) == 1:
                final_hashtags.append(group[0])
            else:
                # Choose the best hashtag from the group
                best_hashtag = self._select_best_hashtag(group)
                final_hashtags.append(best_hashtag)

        return final_hashtags

    def _select_best_hashtag(self, hashtag_group: List[str]) -> str:
        """Select the best hashtag from a group of similar hashtags."""
        if len(hashtag_group) == 1:
            return hashtag_group[0]

        # Score each hashtag
        scored_hashtags = []

        for hashtag in hashtag_group:
            score = 0.0
            content = hashtag[1:]  # Remove #

            # Prefer hashtags with both Persian and English content
            has_persian = bool(self.persian_pattern.search(content))
            has_english = bool(self.english_pattern.search(content))

            if has_persian and has_english:
                score += 0.3
            elif has_persian:
                score += 0.2
            elif has_english:
                score += 0.1

            # Prefer reasonable length
            if 3 <= len(content) <= 25:
                score += 0.2

            # Prefer hashtags with numbers (like mazda3, mx30)
            if any(c.isdigit() for c in content):
                score += 0.1

            # Prefer hashtags without too many special characters
            special_count = content.count('_') + content.count(self.zwj) + content.count(self.zwnj)
            if special_count <= 2:
                score += 0.1

            scored_hashtags.append((hashtag, score))

        # Return the highest scored hashtag
        scored_hashtags.sort(key=lambda x: x[1], reverse=True)
        return scored_hashtags[0][0]

    async def extract_username_from_image(self, image_path: str) -> Optional[str]:
        """Enhanced username extraction with multiple strategies."""
        try:
            start_time = time.time()

            # Strategy 1: English-only extraction optimized for usernames
            username_task = self._extract_text_with_strategy(
                image_path, 'eng', 'english', 'username'
            )

            # Strategy 2: General extraction as fallback
            general_task = self._extract_text_with_strategy(
                image_path, 'eng+fas', 'mixed', 'standard'
            )

            # Execute strategies concurrently
            username_text, general_text = await asyncio.gather(username_task, general_task)

            # Combine texts for better coverage
            combined_text = f"{username_text}\n{general_text}"

            # Extract username using prioritized patterns
            username = self._extract_username_from_text(combined_text)

            extraction_time = time.time() - start_time
            logger.info(f"Username extraction took {extraction_time:.2f}s, result: {username}")

            return username

        except Exception as e:
            logger.error(f"Username extraction failed: {e}")
            return None

    def _extract_username_from_text(self, text: str) -> Optional[str]:
        """Extract username using prioritized pattern matching."""
        if not text:
            return None

        try:
            # Process text line by line for better accuracy
            lines = text.split('\n')
            candidates = []

            for pattern in self.username_patterns:
                for line in lines:
                    matches = pattern.findall(line.strip())
                    for match in matches:
                        if self._is_valid_instagram_username(match):
                            candidates.append((match.lower(), self._score_username(match, line)))

            # Return the highest scored username
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]

            return None

        except Exception as e:
            logger.warning(f"Username pattern matching failed: {e}")
            return None

    def _score_username(self, username: str, context: str) -> float:
        """Score username candidate based on context and validity."""
        score = 0.0

        # Base score for valid format
        score += 0.5

        # Bonus for @ prefix in context
        if f'@{username}' in context:
            score += 0.3

        # Bonus for reasonable length
        if 3 <= len(username) <= 20:
            score += 0.2
        elif len(username) <= 30:
            score += 0.1

        # Penalty for too many dots/underscores
        special_chars = username.count('.') + username.count('_')
        if special_chars > 2:
            score -= 0.2

        # Bonus for alphanumeric content
        if any(c.isalnum() for c in username):
            score += 0.1

        return score

    @lru_cache(maxsize=100)
    def _is_valid_instagram_username(self, username: str) -> bool:
        """Cached validation for Instagram username format."""
        if not username or len(username) > 30 or len(username) < 1:
            return False

        # Check for invalid patterns
        if (username.startswith('.') or username.endswith('.') or
            '..' in username or '__' in username):
            return False

        # Character validation
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._')
        if not all(c in allowed_chars for c in username):
            return False

        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in username):
            return False

        # Exclude common false positives
        excluded_terms = {'www', 'http', 'https', 'com', 'instagram', 'post', 'reel',
                         'story', 'stories', 'follow', 'like', 'share'}
        if username.lower() in excluded_terms:
            return False

        return True

    async def analyze_screenshot_with_ocr(self, image_path: str,
                                        content_type: ContentType) -> Optional[ExtractedData]:
        """Optimized screenshot analysis with concurrent extraction."""
        try:
            start_time = time.time()

            # Extract hashtags and username concurrently
            hashtags_task = self.extract_hashtags_from_image(image_path)
            username_task = self.extract_username_from_image(image_path)

            hashtags, username = await asyncio.gather(hashtags_task, username_task)

            total_time = time.time() - start_time

            # Enhanced logging with language breakdown
            if hashtags:
                persian_hashtags = [h for h in hashtags if self.persian_pattern.search(h)]
                english_hashtags = [h for h in hashtags if not self.persian_pattern.search(h)]
                logger.info(f"OCR Results - Username: {username}")
                logger.info(f"Persian hashtags ({len(persian_hashtags)}): {persian_hashtags}")
                logger.info(f"English hashtags ({len(english_hashtags)}): {english_hashtags}")
            else:
                logger.info(f"OCR Results - Username: {username}, No hashtags found")

            logger.info(f"Total OCR analysis completed in {total_time:.2f}s")

            # Return data if we found something useful
            if hashtags or username:
                confidence = self._calculate_confidence(hashtags, username)

                return ExtractedData(
                    username=username,
                    hashtags=hashtags,
                    content_type=content_type,
                    confidence=confidence,
                    extraction_method="ocr_multilingual_v3"
                )

            logger.info("OCR found no useful data")
            return None

        except Exception as e:
            logger.error(f"OCR analysis failed: {e}")
            return None

    def _calculate_confidence(self, hashtags: List[str], username: Optional[str]) -> float:
        """Calculate confidence score for extraction results."""
        confidence = 0.0

        if hashtags:
            confidence += 0.4  # Base confidence for hashtags
            confidence += min(0.2, len(hashtags) * 0.04)  # Bonus for multiple hashtags

            # Higher confidence for Persian hashtags (harder to extract)
            persian_count = sum(1 for h in hashtags if self.persian_pattern.search(h))
            if persian_count > 0:
                confidence += min(0.15, persian_count * 0.05)

            # Bonus for mixed language hashtags
            english_count = len(hashtags) - persian_count
            if persian_count > 0 and english_count > 0:
                confidence += 0.1

        if username:
            confidence += 0.3  # Base confidence for username
            # Bonus for high-quality usernames
            if 3 <= len(username) <= 20 and username.count('.') + username.count('_') <= 1:
                confidence += 0.1

        return min(0.85, confidence)  # Cap OCR confidence

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)