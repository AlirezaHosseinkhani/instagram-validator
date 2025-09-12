"""
Service for scraping Instagram content using instaloader.
"""

import asyncio
import instaloader
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse
from app.utils.exceptions import InstagramScrapingError
from app.models.database import ContentType
from app.models.api import ExtractedData
import logging

logger = logging.getLogger(__name__)


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

    def shortcode_from_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract shortcode and content type from Instagram URL.

        Args:
            url: Instagram URL

        Returns:
            Tuple[Optional[str], Optional[str]]: (shortcode, content_type)
        """
        try:
            path = urlparse(url).path.strip("/")
            parts = path.split("/")
            if len(parts) >= 2 and parts[0] in ("p", "reel", "tv"):
                return parts[1], parts[0]  # return shortcode and type
            return None, None
        except Exception as e:
            logger.warning(f"Failed to extract shortcode from URL {url}: {str(e)}")
            return None, None

    async def get_post_info(self, post_url: str, user: str = None, password: str = None) -> Optional[Dict[str, Any]]:
        """
        Get post information using instaloader.

        Args:
            post_url: Instagram post URL
            user: Optional username for login
            password: Optional password for login

        Returns:
            Optional[Dict[str, Any]]: Post information or None if failed
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._get_post_info_sync,
                post_url,
                user,
                password
            )
        except Exception as e:
            logger.warning(f"Failed to get post info: {str(e)}")
            return None

    def _get_post_info_sync(self, post_url: str, user: str = None, password: str = None) -> Optional[Dict[str, Any]]:
        """Synchronous post info retrieval."""
        try:
            shortcode, ctype = self.shortcode_from_url(post_url)
            if not shortcode:
                return None

            L = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_geotags=False,
                download_comments=False,
                save_metadata=False,
                compress_json=False
            )

            if user and password:   # login if you want private account access
                L.login(user, password)

            post = instaloader.Post.from_shortcode(L.context, shortcode)
            profile = post.owner_profile

            content_type = {
                "p": "post",
                "reel": "reel",
                "tv": "igtv"
            }.get(ctype, "unknown")

            # Extract hashtags from caption
            hashtags = []
            if post.caption:
                import re
                hashtag_pattern = r'#[a-zA-Z0-9_]+'
                hashtags = re.findall(hashtag_pattern, post.caption)
                hashtags = [tag.lower() for tag in hashtags]

            return {
                "username": post.owner_username,
                "is_private": profile.is_private,
                "is_verified": profile.is_verified,
                "content_type": content_type,
                "caption": post.caption,
                "hashtags": hashtags,
                "mentions": list(post.caption_mentions) if hasattr(post, 'caption_mentions') else [],
            }

        except instaloader.exceptions.PostUnavailableException:
            logger.warning("Post is not available (private or deleted)")
            return None
        except instaloader.exceptions.ProfileNotExistsException:
            logger.warning("Profile does not exist")
            return None
        except instaloader.exceptions.LoginRequiredException:
            logger.warning("Login required to access this content")
            return None
        except Exception as e:
            logger.warning(f"Failed to scrape post: {str(e)}")
            return None

    async def scrape_post_data(self, post_url: str) -> Optional[ExtractedData]:
        """
        Scrape data from Instagram post URL using new method.

        Args:
            post_url: Instagram post URL

        Returns:
            Optional[ExtractedData]: Extracted post data or None if failed
        """
        try:
            # Try to get post info using the new method
            post_info = await self.get_post_info(post_url)

            if not post_info:
                logger.info("Instaloader method failed to extract post info")
                return None

            # Convert content type string to ContentType enum
            content_type_map = {
                "post": ContentType.POST,
                "reel": ContentType.REELS,
                "igtv": ContentType.POST,  # Treat IGTV as post for now
                "unknown": ContentType.POST
            }

            content_type = content_type_map.get(post_info.get("content_type", "unknown"), ContentType.POST)

            logger.info(f"Instaloader successfully extracted data for user: {post_info['username']}")

            return ExtractedData(
                username=post_info["username"],
                hashtags=post_info["hashtags"],
                content_type=content_type,
                confidence=0.95,  # High confidence for scraped data
                extraction_method="instaloader_shortcode"
            )

        except Exception as e:
            logger.warning(f"Failed to scrape post data: {str(e)}")
            return None