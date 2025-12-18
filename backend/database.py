"""
Database layer for CognitiveAI with User model.
Handles user persistence and authentication.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import json

logger = logging.getLogger(__name__)

DATABASE_PATH = Path(__file__).parent.parent / "cognitiveai.db"


def get_connection(db_path: str = str(DATABASE_PATH)) -> sqlite3.Connection:
    """Create a sqlite3 connection configured for WAL mode and reasonable timeouts."""
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        
        pass
    return conn


class User:
    """User data model."""
    def __init__(self, user_id: str, username: str, password_hash: str, email: Optional[str] = None, created_at: Optional[str] = None):
        self.user_id = user_id
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.created_at = created_at or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Return user data as dict (without password hash)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at
        }


class Database:
    """SQLite database manager for users."""

    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database schema."""
        with get_connection(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pdf_documents (
                    document_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT,
                    title TEXT,
                    file_size INTEGER,
                    upload_timestamp REAL,
                    metadata TEXT
                )
            ''')

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def create_user(self, user_id: str, username: str, password_hash: str, email: Optional[str] = None) -> User:
        """Create a new user in the database."""
        user = User(user_id, username, password_hash, email)
        try:
            with get_connection(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO users (user_id, username, password_hash, email, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user.user_id, user.username, user.password_hash, user.email, user.created_at)
                )
                conn.commit()
            logger.info(f"User created: {username}")
            return user
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Retrieve user by username."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT user_id, username, password_hash, email, created_at FROM users WHERE username = ?",
                (username,)
            ).fetchone()
        
        if row:
            return User(row[0], row[1], row[2], row[3], row[4])
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Retrieve user by user_id."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT user_id, username, password_hash, email, created_at FROM users WHERE user_id = ?",
                (user_id,)
            ).fetchone()
        
        if row:
            return User(row[0], row[1], row[2], row[3], row[4])
        return None

    def username_exists(self, username: str) -> bool:
        """Check if username already exists."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (username,)
            ).fetchone()
        return row is not None

    def delete_user(self, user_id: str) -> bool:
        """Delete a user (for cleanup/testing)."""
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    
    def create_pdf_metadata(self, document_id: str, user_id: str, filename: str, title: str, file_size: int, upload_timestamp: float, metadata: Optional[Dict[str, Any]] = None):
        """Create or update PDF metadata entry."""
        meta_json = json.dumps(metadata or {})
        with get_connection(self.db_path) as conn:
            conn.execute(
                "REPLACE INTO pdf_documents (document_id, user_id, filename, title, file_size, upload_timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, user_id, filename, title, file_size, upload_timestamp, meta_json)
            )
            conn.commit()

    def get_pdf_documents_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve PDF metadata entries for a given user."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT document_id, filename, title, file_size, upload_timestamp, metadata FROM pdf_documents WHERE user_id = ?",
                (user_id,)
            ).fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            try:
                meta = json.loads(r[5]) if r[5] else {}
            except Exception:
                meta = {}
            results.append({
                "document_id": r[0],
                "filename": r[1],
                "title": r[2],
                "file_size": r[3],
                "upload_timestamp": r[4],
                "metadata": meta
            })
        return results

    def delete_pdf_metadata(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Delete PDF metadata entry (scoped to user if provided)."""
        with get_connection(self.db_path) as conn:
            if user_id:
                cursor = conn.execute("DELETE FROM pdf_documents WHERE document_id = ? AND user_id = ?", (document_id, user_id))
            else:
                cursor = conn.execute("DELETE FROM pdf_documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return cursor.rowcount > 0



db: Optional[Database] = None


def get_database() -> Database:
    """Get or initialize the global database instance."""
    global db
    if db is None:
        db = Database()
    return db
