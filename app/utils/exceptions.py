"""
Custom exceptions for the Instagram Validator application.
"""

class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

class InstagramScrapingError(ValidationError):
    """Exception raised when Instagram scraping fails."""
    pass

class OpenAIAnalysisError(ValidationError):
    """Exception raised when OpenAI analysis fails."""
    pass

class URLParsingError(ValidationError):
    """Exception raised when URL parsing fails."""
    pass

class FileValidationError(ValidationError):
    """Exception raised when file validation fails."""
    pass

class DatabaseError(Exception):
    """Exception raised for database operations."""
    pass

class AuthenticationError(Exception):
    """Exception raised for authentication failures."""
    pass