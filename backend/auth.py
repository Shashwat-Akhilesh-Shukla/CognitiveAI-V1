"""
Authentication module for CognitiveAI.
Handles JWT token generation, verification, and password management.

Auth model: access tokens only. No refresh tokens are issued or supported by this service.
Clients must discard access tokens on logout.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import uuid

import jwt
import bcrypt

logger = logging.getLogger(__name__)


SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = int(os.getenv("TOKEN_EXPIRY_HOURS", "24"))


class AuthService:
    """Handles authentication: tokens, passwords, user validation."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

    @staticmethod
    def generate_token(user_id: str, username: str) -> str:
        """Generate a JWT token for a user."""
        payload = {
            "user_id": user_id,
            "username": username,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return token

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token. Returns payload if valid, None otherwise."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    @staticmethod
    def generate_user_id() -> str:
        """Generate a unique user_id (UUID)."""
        return str(uuid.uuid4())

    @staticmethod
    def validate_username(username: str) -> tuple[bool, Optional[str]]:
        """Validate username format. Returns (is_valid, error_message)."""
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        if len(username) > 50:
            return False, "Username must be 50 characters or less"
        if not username.isalnum() and '_' not in username and '-' not in username:
            return False, "Username can only contain letters, numbers, underscores, and hyphens"
        return True, None

    @staticmethod
    def validate_password(password: str) -> tuple[bool, Optional[str]]:
        """Validate password strength. Returns (is_valid, error_message)."""
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters"
        if len(password) > 128:
            return False, "Password must be 128 characters or less"
        return True, None
