"""
Main FastAPI application module.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os

from app.config import settings
from app.database import create_db_and_tables
from app.api.endpoints import router as api_router
from app.utils.exceptions import ValidationError, FileValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Instagram Validator API...")

    # Create database tables
    create_db_and_tables()
    logger.info("Database tables created/verified")

    # Verify configuration
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured - LLM analysis will fail")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Instagram Validator API...")

# Create FastAPI application
app = FastAPI(
    title="Instagram Marketing Campaign Validator",
    description="""
    A robust FastAPI application for validating Instagram content submissions for marketing campaigns.
    
    ## Features
    
    * **Content Submission**: Submit Instagram posts, stories, or reels for validation
    * **Automated Validation**: Extract usernames and hashtags using scraping or AI analysis
    * **Account Visibility**: Detect public vs private Instagram accounts
    * **Hashtag Compliance**: Verify required campaign hashtags are present
    * **Admin Panel**: Secure admin interface for managing submissions
    * **Real-time Status**: Track validation progress and results
    
    ## Validation Workflow
    
    1. **URL Parsing**: Extract username from Instagram URL
    2. **Account Check**: Determine if account is public or private
    3. **Content Analysis**: Scrape public content or analyze screenshot with AI
    4. **Username Validation**: Verify URL username matches content username
    5. **Hashtag Validation**: Check for required campaign hashtags
    
    ## Admin Access
    
    Access the admin panel at `/admin` with configured credentials.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Setup admin panel (import here to avoid circular imports)
from app.api.admin import setup_admin
admin = setup_admin(app)

# Serve uploaded files (for development)
if settings.debug and os.path.exists(settings.upload_dir):
    app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# Global exception handlers (moved from endpoints.py)
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": "Validation Error", "detail": str(exc)}
    )

@app.exception_handler(FileValidationError)
async def file_validation_exception_handler(request: Request, exc: FileValidationError):
    """Handle file validation errors."""
    logger.warning(f"File validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": "File Validation Error", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(exc)}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Instagram Marketing Campaign Validator API",
        "version": "1.0.0",
        "docs": "/docs",
        "admin": "/admin",
        "status": "running"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )