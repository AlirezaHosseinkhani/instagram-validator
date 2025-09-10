"""
Service for parsing Instagram URLs and extracting usernames.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse
from app.utils.exceptions import URLParsingError
from app.models.database import ContentType


class URLParser:
    """Service for parsing Instagram URLs."""

    @staticmethod
    def extract_username_from_url(url: str) -> str:
        """
        Extract username from Instagram URL.

        Args:
            url: Instagram URL

        Returns:
            str: Extracted username

        Raises:
            URLParsingError: If username cannot be extracted
        """
        try:
            parsed_url = urlparse(url)
            if 'instagram.com' not in parsed_url.netloc:
                raise URLParsingError("URL is not an Instagram URL")

            path = parsed_url.path.strip('/')

            # Pattern for different Instagram URL types
            patterns = [
                # Stories: /stories/username/story_id
                (r'^stories/([^/]+)/\d+', ContentType.STORY),
                # Posts: /p/post_id or /username (profile)
                (r'^p/[^/]+', None),  # Will need to scrape for username
                # Reels: /reel/reel_id
                (r'^reel/[^/]+', None),  # Will need to scrape for username
                # Profile: /username
                (r'^([^/]+)/?$', None),
            ]

            for pattern, content_type in patterns:
                match = re.match(pattern, path)
                if match and content_type == ContentType.STORY:
                    # For stories, we can directly extract username
                    return match.group(1)
                elif match and pattern.startswith('^([^/]+)'):
                    # For profile URLs, extract username directly
                    return match.group(1)

            # If we can't extract username directly, we'll need to scrape
            return None

        except Exception as e:
            raise URLParsingError(f"Failed to parse Instagram URL: {str(e)}")

    @staticmethod
    def determine_content_type_from_url(url: str) -> Optional[ContentType]:
        """
        Determine content type from Instagram URL.

        Args:
            url: Instagram URL

        Returns:
            Optional[ContentType]: Detected content type or None
        """
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.strip('/')

            if path.startswith('stories/'):
                return ContentType.STORY
            elif path.startswith('p/'):
                return ContentType.POST
            elif path.startswith('reel/'):
                return ContentType.REELS

            return None

        except Exception:
            return None

    @staticmethod
    def validate_instagram_url(url: str) -> Tuple[bool, str]:
        """
        Validate Instagram URL format.

        Args:
            url: URL to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            parsed_url = urlparse(url)

            # Check if it's an Instagram URL
            if 'instagram.com' not in parsed_url.netloc:
                return False, "URL must be an Instagram URL"

            # Check if URL has a valid path
            if not parsed_url.path or parsed_url.path == '/':
                return False, "Instagram URL must contain content path"

            path = parsed_url.path.strip('/')

            # Valid Instagram URL patterns
            valid_patterns = [
                r'^stories/[^/]+/\d+',  # Stories
                r'^p/[A-Za-z0-9_-]+',  # Posts
                r'^reel/[A-Za-z0-9_-]+',  # Reels
                r'^[A-Za-z0-9_.]+/?$',  # Profile
            ]

            if not any(re.match(pattern, path) for pattern in valid_patterns):
                return False, "Invalid Instagram URL format"

            return True, ""

        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"