"""
Service for scraping Instagram content using instaloader.
"""

import asyncio
import instaloader
from typing import Optional, Dict, List, Any
from datetime import datetime
import tempfile
import os
from app.utils.exceptions import InstagramScrapingError
from app.models.database import ContentType
from app.models.api import ExtractedData


class InstagramScraper:
    """Service for scraping Instagram content."""

    def __init__(self):
        """Initialize Instagram scraper."""
        self.loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False
        )

    async def check_account_visibility(self, username: str) -> bool:
        """
        Check if Instagram account is public or private.

        Args:
            username: Instagram username

        Returns:
            bool: True if public, False if private

        Raises:
            InstagramScrapingError: If account check fails
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._check_account_visibility_sync,
                username
            )
        except Exception as e:
            raise InstagramScrapingError(f"Failed to check account visibility: {str(e)}")

    def _check_account_visibility_sync(self, username: str) -> bool:
        """Synchronous account visibility check."""
        try:
            profile = instaloader.Profile.from_username(self.loader.context, username)
            return not profile.is_private
        except instaloader.exceptions.ProfileNotExistsException:
            raise InstagramScrapingError(f"Instagram account '{username}' does not exist")
        except Exception as e:
            raise InstagramScrapingError(f"Failed to access profile: {str(e)}")

    async def scrape_post_data(self, post_url: str) -> ExtractedData:
        """
        Scrape data from Instagram post URL.

        Args:
            post_url: Instagram post URL

        Returns:
            ExtractedData: Extracted post data

        Raises:
            InstagramScrapingError: If scraping fails
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._scrape_post_data_sync,
                post_url
            )
        except Exception as e:
            raise InstagramScrapingError(f"Failed to scrape post data: {str(e)}")

    def _scrape_post_data_sync(self, post_url: str) -> ExtractedData:
        """Synchronous post data scraping."""
        try:
            # Extract shortcode from URL
            shortcode = self._extract_shortcode_from_url(post_url)
            if not shortcode:
                raise InstagramScrapingError("Could not extract shortcode from URL")

            # Get post data
            post = instaloader.Post.from_shortcode(self.loader.context, shortcode)

            # Extract hashtags from caption
            hashtags = []
            if post.caption:
                import re
                hashtag_pattern = r'#[a-zA-Z0-9_]+'
                hashtags = re.findall(hashtag_pattern, post.caption)
                hashtags = [tag.lower() for tag in hashtags]

            # Determine content type
            content_type = ContentType.POST
            if post.is_video and hasattr(post, 'video_duration'):
                # Check if it's a reel (typically shorter videos)
                if post.video_duration and post.video_duration < 90:  # Reels are usually under 90 seconds
                    content_type = ContentType.REELS

            return ExtractedData(
                username=post.owner_username,
                hashtags=hashtags,
                content_type=content_type,
                confidence=0.95,  # High confidence for scraped data
                extraction_method="scraping"
            )

        except instaloader.exceptions.PostUnavailableException:
            raise InstagramScrapingError("Post is not available (private or deleted)")
        except instaloader.exceptions.ProfileNotExistsException:
            raise InstagramScrapingError("Profile does not exist")
        except Exception as e:
            raise InstagramScrapingError(f"Failed to scrape post: {str(e)}")

    def _extract_shortcode_from_url(self, url: str) -> Optional[str]:
        """Extract shortcode from Instagram URL."""
        import re

        # Pattern for post URLs: /p/shortcode/
        post_pattern = r'/p/([A-Za-z0-9_-]+)'
        match = re.search(post_pattern, url)
        if match:
            return match.group(1)

        # Pattern for reel URLs: /reel/shortcode/
        reel_pattern = r'/reel/([A-Za-z0-9_-]+)'
        match = re.search(reel_pattern, url)
        if match:
            return match.group(1)

        return None

    async def get_profile_info(self, username: str) -> Dict[str, Any]:
        """
        Get basic profile information.

        Args:
            username: Instagram username

        Returns:
            Dict[str, Any]: Profile information

        Raises:
            InstagramScrapingError: If profile access fails
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._get_profile_info_sync,
                username
            )
        except Exception as e:
            raise InstagramScrapingError(f"Failed to get profile info: {str(e)}")

    def _get_profile_info_sync(self, username: str) -> Dict[str, Any]:
        """Synchronous profile info retrieval."""
        try:
            profile = instaloader.Profile.from_username(self.loader.context, username)

            return {
                "username": profile.username,
                "full_name": profile.full_name,
                "is_private": profile.is_private,
                "is_verified": profile.is_verified,
                "followers_count": profile.followers,
                "following_count": profile.followees,
                "posts_count": profile.mediacount,
                "biography": profile.biography
            }

        except instaloader.exceptions.ProfileNotExistsException:
            raise InstagramScrapingError(f"Profile '{username}' does not exist")
        except Exception as e:
            raise InstagramScrapingError(f"Failed to access profile: {str(e)}")