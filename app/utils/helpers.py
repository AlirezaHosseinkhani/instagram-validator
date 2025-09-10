"""
Helper utilities for file handling and data processing.
"""

import os
import re
import aiofiles
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from app.config import settings
from app.utils.exceptions import FileValidationError


async def save_uploaded_file(file: UploadFile, upload_dir: str = None) -> str:
    """
    Save uploaded file to disk and return the file path.

    Args:
        file: FastAPI UploadFile object
        upload_dir: Directory to save the file (defaults to settings.upload_dir)

    Returns:
        str: Path to the saved file

    Raises:
        FileValidationError: If file validation fails
    """
    if upload_dir is None:
        upload_dir = settings.upload_dir

    # Validate file type and size
    validate_file_type(file)
    await validate_file_size(file)

    # Generate unique filename
    import uuid
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)

    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return file_path


def validate_file_type(file: UploadFile) -> None:
    """
    Validate file type against allowed types.

    Args:
        file: FastAPI UploadFile object

    Raises:
        FileValidationError: If file type is not allowed
    """
    if file.content_type not in settings.allowed_file_types:
        raise FileValidationError(
            f"File type {file.content_type} not allowed. "
            f"Allowed types: {', '.join(settings.allowed_file_types)}"
        )


async def validate_file_size(file: UploadFile) -> None:
    """
    Validate file size against maximum allowed size.

    Args:
        file: FastAPI UploadFile object

    Raises:
        FileValidationError: If file is too large
    """
    # Read file to check size
    content = await file.read()
    file_size = len(content)

    # Reset file pointer
    await file.seek(0)

    if file_size > settings.max_file_size:
        raise FileValidationError(
            f"File size {file_size} bytes exceeds maximum allowed size "
            f"{settings.max_file_size} bytes"
        )


def clean_hashtag(hashtag: str) -> str:
    """
    Clean and normalize hashtag.

    Args:
        hashtag: Raw hashtag string

    Returns:
        str: Cleaned hashtag
    """
    # Remove extra whitespace and ensure hashtag starts with #
    hashtag = hashtag.strip()
    if not hashtag.startswith('#'):
        hashtag = f"#{hashtag}"

    # Convert to lowercase for comparison
    return hashtag.lower()


def normalize_username(username: str) -> str:
    """
    Normalize Instagram username.

    Args:
        username: Raw username string

    Returns:
        str: Normalized username
    """
    # Remove @ symbol if present and convert to lowercase
    username = username.strip().lower()
    if username.startswith('@'):
        username = username[1:]

    return username


def extract_hashtags_from_text(text: str) -> List[str]:
    """
    Extract hashtags from text using regex.

    Args:
        text: Text containing hashtags

    Returns:
        List[str]: List of extracted hashtags
    """
    # Regex pattern to match hashtags
    hashtag_pattern = r'#[a-zA-Z0-9_]+'
    hashtags = re.findall(hashtag_pattern, text)

    # Clean and normalize hashtags
    return [clean_hashtag(tag) for tag in hashtags]


def validate_instagram_url(url: str) -> bool:
    """
    Validate if URL is a valid Instagram URL.

    Args:
        url: URL string to validate

    Returns:
        bool: True if valid Instagram URL
    """
    instagram_patterns = [
        r'https?://(?:www\.)?instagram\.com/p/[\w-]+/?',
        r'https?://(?:www\.)?instagram\.com/stories/[\w.-]+/\d+/?',
        r'https?://(?:www\.)?instagram\.com/reel/[\w-]+/?',
        r'https?://(?:www\.)?instagram\.com/[\w.-]+/?'
    ]

    return any(re.match(pattern, url) for pattern in instagram_patterns)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"