"""
Admin panel configuration using SQLAdmin.
"""

from sqlalchemy import create_engine
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import RedirectResponse
from app.config import settings
from app.models.database import Submission, ValidationResult
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AdminAuth(AuthenticationBackend):
    """Authentication backend for admin panel."""

    async def login(self, request: Request) -> bool:
        """Handle admin login."""
        form = await request.form()
        username, password = form["username"], form["password"]

        # Verify credentials
        if username == settings.admin_username and password == settings.admin_password:
            # Create JWT token
            token_data = {
                "sub": username,
                "exp": datetime.utcnow() + timedelta(hours=24)
            }
            token = jwt.encode(token_data, settings.admin_secret_key, algorithm="HS256")

            # Set token in session
            request.session.update({"token": token})
            return True

        return False

    async def logout(self, request: Request) -> bool:
        """Handle admin logout."""
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        """Authenticate admin user."""
        token = request.session.get("token")

        if not token:
            return False

        try:
            payload = jwt.decode(token, settings.admin_secret_key, algorithms=["HS256"])
            username: str = payload.get("sub")
            if username is None:
                return False

            # Check if user is admin
            return username == settings.admin_username

        except JWTError:
            return False

class SubmissionAdmin(ModelView, model=Submission):
    """Admin view for submissions."""

    column_list = [
        Submission.id,
        Submission.instagram_url,
        Submission.content_type,
        Submission.validation_status,
        Submission.url_username,
        Submission.content_username,
        Submission.username_match,
        Submission.hashtags_valid,
        Submission.created_at,
        Submission.validated_at
    ]

    column_searchable_list = [
        Submission.instagram_url,
        Submission.url_username,
        Submission.content_username
    ]

    column_filters = [
        Submission.validation_status,
        Submission.content_type,
        Submission.username_match,
        Submission.hashtags_valid,
        Submission.is_account_public
    ]

    column_sortable_list = [
        Submission.id,
        Submission.created_at,
        Submission.validated_at,
        Submission.validation_status
    ]

    # Make certain fields read-only
    form_excluded_columns = [
        Submission.id,
        Submission.created_at,
        Submission.validated_at
    ]

    # Custom column labels
    column_labels = {
        Submission.instagram_url: "Instagram URL",
        Submission.content_type: "Content Type",
        Submission.validation_status: "Status",
        Submission.url_username: "URL Username",
        Submission.content_username: "Content Username",
        Submission.username_match: "Username Match",
        Submission.hashtags_valid: "Hashtags Valid",
        Submission.is_account_public: "Public Account",
        Submission.created_at: "Created",
        Submission.validated_at: "Validated"
    }

    # Items per page
    page_size = 50
    page_size_options = [25, 50, 100, 200]

class ValidationResultAdmin(ModelView, model=ValidationResult):
    """Admin view for validation results."""

    column_list = [
        ValidationResult.id,
        ValidationResult.submission_id,
        ValidationResult.url_parsing_success,
        ValidationResult.account_check_success,
        ValidationResult.content_extraction_success,
        ValidationResult.username_validation_success,
        ValidationResult.hashtag_validation_success,
        ValidationResult.extraction_method,
        ValidationResult.extraction_confidence,
        ValidationResult.created_at
    ]

    column_filters = [
        ValidationResult.url_parsing_success,
        ValidationResult.account_check_success,
        ValidationResult.content_extraction_success,
        ValidationResult.username_validation_success,
        ValidationResult.hashtag_validation_success,
        ValidationResult.extraction_method
    ]

    column_sortable_list = [
        ValidationResult.id,
        ValidationResult.submission_id,
        ValidationResult.extraction_confidence,
        ValidationResult.created_at
    ]

    # Make all fields read-only (this is audit data)
    can_create = False
    can_edit = False
    can_delete = False

    # Custom column labels
    column_labels = {
        ValidationResult.submission_id: "Submission ID",
        ValidationResult.url_parsing_success: "URL Parsing",
        ValidationResult.account_check_success: "Account Check",
        ValidationResult.content_extraction_success: "Content Extraction",
        ValidationResult.username_validation_success: "Username Validation",
        ValidationResult.hashtag_validation_success: "Hashtag Validation",
        ValidationResult.extraction_method: "Extraction Method",
        ValidationResult.extraction_confidence: "Confidence",
        ValidationResult.created_at: "Created"
    }

    # Items per page
    page_size = 50

def setup_admin(app):
    """Setup admin panel with the FastAPI app."""
    # Create admin authentication
    authentication_backend = AdminAuth(secret_key=settings.admin_secret_key)

    # Create admin instance with the app
    admin = Admin(
        app=app,
        engine=create_engine(settings.database_url),
        authentication_backend=authentication_backend,
        title="Instagram Validator Admin",
        base_url="/admin"
    )

    # Add views to admin
    admin.add_view(SubmissionAdmin)
    admin.add_view(ValidationResultAdmin)

    return admin