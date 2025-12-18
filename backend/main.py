"""
FastAPI Backend for CognitiveAI with Multi-User Support

Provides REST API endpoints for authentication, chat, PDF upload, and memory inspection.
All data is strictly isolated per user.
"""

import os
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import uvicorn
import logging
from dotenv import load_dotenv
import time

load_dotenv("backend/.env")

from backend.memory.stm import STMManager
from backend.memory.ltm import LTMManager
from backend.pdf_loader import PDFLoader
from backend.reasoning import CognitiveReasoningEngine
from backend.database import get_database, Database, User
from backend.auth import AuthService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="CognitiveAI API",
    description="Memory-Augmented Personal Intelligence Engine (Multi-User)",
    version="2.0.0"
)


frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
allow_origins = [frontend_url]
if frontend_url != "http://localhost:3000":
    
    allow_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["*"],
)


stm_manager: Optional[STMManager] = None
ltm_manager: Optional[LTMManager] = None
pdf_loader: Optional[PDFLoader] = None
reasoning_engine: Optional[CognitiveReasoningEngine] = None
db: Optional[Database] = None









class SignupRequest(BaseModel):
    """Request model for user signup."""
    username: str
    password: str
    email: Optional[str] = None


class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str
    password: str


class AuthResponse(BaseModel):
    """Response model for auth endpoints."""
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    client_discard_token: Optional[bool] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    doc_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    reasoning: Dict[str, Any]
    metadata: Dict[str, Any]


class MemoryStats(BaseModel):
    """Response model for memory statistics."""
    stm_count: int
    ltm_stats: Dict[str, Any]
    pdf_documents: List[Dict[str, Any]]
    reasoning_stats: Dict[str, Any]






def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract and verify user_id from JWT token in Authorization header.
    Raises HTTPException if token is invalid or missing.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")

    try:
        
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise ValueError("Invalid authorization header format")

        token = parts[1]
        payload = AuthService.verify_token(token)

        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing user_id")

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")






def validate_environment():
    """Validate required environment variables at startup."""
    required_vars = ["JWT_SECRET_KEY", "REDIS_URL", "PERPLEXITY_API_KEY"]
    missing = []
    for var in required_vars:
        val = os.getenv(var)
        if not val:
            missing.append(var)
        elif var == "JWT_SECRET_KEY" and val == "your-secret-key-change-in-production":
            raise RuntimeError(f"Environment variable {var} is set to default placeholder; must change in production")

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    logger.info("✓ Environment validation passed")


def initialize_memory_systems():
    """Initialize global memory systems and reasoning engine."""
    global stm_manager, ltm_manager, pdf_loader, reasoning_engine, db

    try:
        jwt_secret = os.getenv("JWT_SECRET_KEY")
        if not jwt_secret or jwt_secret == "your-secret-key-change-in-production":
            raise RuntimeError("JWT_SECRET_KEY is required and must be set to a secure value")

        
        db = get_database()

        
        
        redis_url = os.getenv("REDIS_URL")
        stm_ttl = int(os.getenv("STM_TTL", "1800"))
        if not redis_url:
            raise RuntimeError("REDIS_URL is required. STM must use Redis only.")

        stm_manager = STMManager(redis_url=redis_url, ttl_seconds=stm_ttl, max_size=50)

        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            logger.warning("PINECONE_API_KEY not set — LTM disabled.")
            ltm_manager = None
        else:
            try:
                ltm_manager = LTMManager(
                    api_key=pinecone_api_key,
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                    index_name="cognitiveai-ltm"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LTM: {e}")
                ltm_manager = None

        
        pdf_loader = PDFLoader(ltm_manager)

        
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if not perplexity_api_key:
            logger.warning("PERPLEXITY_API_KEY not set. Chat will not work.")

        reasoning_engine = CognitiveReasoningEngine(
            stm_manager=stm_manager,
            ltm_manager=ltm_manager,
            pdf_loader=pdf_loader,
            perplexity_api_key=perplexity_api_key or ""
        )

        logger.info("Memory systems initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize memory systems: {e}")
        logger.warning("Continuing without full initialization.")


@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup with env validation."""
    try:
        validate_environment()
        initialize_memory_systems()
        logger.info("✓ Startup complete: All systems initialized and validated")
    except RuntimeError as e:
        logger.critical(f"✗ Startup failed: {e}")
        raise
    except Exception as e:
        logger.critical(f"✗ Unexpected startup error: {e}")
        raise






@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "CognitiveAI API (Multi-User)", "status": "running"}


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with service-level diagnostics."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    
    health_status["services"]["database"] = {"available": db is not None, "status": "ok" if db else "unavailable"}

    
    redis_health = {"available": stm_manager is not None, "status": "unknown"}
    if stm_manager:
        try:
            from backend.redis_client import get_redis
            r = get_redis()
            r.ping()
            redis_health["status"] = "ok"
        except Exception as e:
            redis_health["status"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    health_status["services"]["redis"] = redis_health

    
    pinecone_health = {"available": ltm_manager is not None, "status": "unknown"}
    if ltm_manager:
        try:
            
            pinecone_health["status"] = "ok"
        except Exception as e:
            pinecone_health["status"] = f"error: {str(e)}"
            
    health_status["services"]["pinecone"] = pinecone_health

    
    perplexity_health = {"available": reasoning_engine is not None and reasoning_engine.perplexity_api_key, "status": "unknown"}
    if perplexity_health["available"]:
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                resp = client.head("https://api.perplexity.ai/chat/completions", headers={"Authorization": f"Bearer {reasoning_engine.perplexity_api_key}"})
                
                perplexity_health["status"] = "ok" if resp.status_code < 500 else "server_error"
        except Exception as e:
            perplexity_health["status"] = f"unreachable: {str(e)}"
            health_status["status"] = "degraded"
    else:
        perplexity_health["status"] = "not_configured"
    health_status["services"]["perplexity"] = perplexity_health

    
    health_status["systems"] = {
        "database": db is not None,
        "stm": stm_manager is not None,
        "ltm": ltm_manager is not None,
        "pdf_loader": pdf_loader is not None,
        "reasoning_engine": reasoning_engine is not None
    }

    return health_status






@app.post("/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    """
    Create a new user account.

    Validates username/password, stores hashed password in DB, returns JWT token.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        
        is_valid_username, username_error = AuthService.validate_username(request.username)
        if not is_valid_username:
            raise HTTPException(status_code=400, detail=username_error)

        is_valid_password, password_error = AuthService.validate_password(request.password)
        if not is_valid_password:
            raise HTTPException(status_code=400, detail=password_error)

        
        if db.username_exists(request.username):
            raise HTTPException(status_code=409, detail="Username already exists")

        
        password_hash = AuthService.hash_password(request.password)

        
        user_id = AuthService.generate_user_id()
        user = db.create_user(
            user_id=user_id,
            username=request.username,
            password_hash=password_hash,
            email=request.email
        )

        

        
        token = AuthService.generate_token(user_id, request.username)

        logger.info(f"New user created: {request.username} (ID: {user_id})")

        return AuthResponse(
            success=True,
            message="User created successfully",
            token=token,
            user=user.to_dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Authenticate a user and return a JWT token.

    Verifies username and password, returns token on success.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        
        user = db.get_user_by_username(request.username)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        
        if not AuthService.verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        

        
        token = AuthService.generate_token(user.user_id, user.username)

        logger.info(f"User logged in: {request.username}")

        return AuthResponse(
            success=True,
            message="Login successful",
            token=token,
            user=user.to_dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/auth/logout", response_model=AuthResponse)
async def logout(user_id: str = Depends(get_current_user)):
    """
    Logout endpoint.

    Clears user's STM and performs cleanup on backend.
    Frontend should discard token after receiving this.
    """
    try:
        
        if stm_manager:
            stm_manager.clear_memories(user_id)

        
        if reasoning_engine:
            try:
                reasoning_engine.clear_short_term_memory_for_user(user_id)
                reasoning_engine.reset_conversation_context_for_user(user_id)
            except Exception:
                
                pass

        
        try:
            from backend.redis_client import get_redis
            r = get_redis()
            
            r.delete(f"conv:{user_id}")
            
            pattern = f"pdf:{user_id}:*"
            for key in r.scan_iter(match=pattern):
                try:
                    r.delete(key)
                except Exception:
                    pass
        except Exception:
            
            pass

        
        if ltm_manager:
            try:
                ltm_manager.delete_memories_by_user(user_id)
            except Exception:
                logger.warning(f"Failed to delete LTM memories for user {user_id}")

        logger.info(f"User logged out: {user_id}")

        return AuthResponse(
            success=True,
            message="Logout successful. Client must discard the access token; no refresh tokens are used.",
            client_discard_token=True
        )

    except Exception as e:
        logger.error(f"Logout error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@app.get("/auth/me")
async def get_current_user_info(user_id: str = Depends(get_current_user)):
    """Get current user's information."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user info")






@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Main chat endpoint (user-scoped).

    Processes user messages through the cognitive reasoning engine.
    All data is isolated to the authenticated user.
    """
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")

    try:
        
        stm_list = []
        try:
            if stm_manager:
                raw_stm = stm_manager.get_relevant_memories(user_id, request.message, limit=5)
                for m in raw_stm:
                    try:
                        stm_list.append({
                            "id": getattr(m, "id", None),
                            "content": getattr(m, "content", str(m)),
                            "timestamp": getattr(m, "timestamp", time.time()),
                            "importance": getattr(m, "importance", 1.0),
                            "metadata": getattr(m, "metadata", {})
                        })
                    except Exception:
                        stm_list.append({"content": str(m)})
        except Exception:
            stm_list = []

        ltm_list = []
        try:
            if ltm_manager:
                ltm_list = ltm_manager.search_memories(request.message, limit=5, user_id=user_id)
        except Exception:
            ltm_list = []

        pdf_snippets = []
        try:
            if pdf_loader:
                if request.doc_id:
                    chunks = pdf_loader.search_pdf_knowledge(query=request.message, document_id=request.doc_id, limit=3, user_id=user_id)
                else:
                    chunks = pdf_loader.search_pdf_knowledge(query=request.message, limit=3, user_id=user_id)
                for c in chunks:
                    content = c.get("content", "")[:300]
                    pdf_snippets.append(content)
        except Exception:
            pdf_snippets = []

        
        result = await reasoning_engine.process_message(
            user_message=request.message,
            user_id=user_id,
            stm_memories=stm_list,
            ltm_memories=ltm_list,
            pdf_snippets=pdf_snippets
        )

        
        try:
            actions = result.get("memory_actions", []) if isinstance(result, dict) else []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                if action.get("type") == "stm" and stm_manager:
                    try:
                        stm_manager.add_memory(user_id, action.get("content", ""), importance=action.get("importance", 0.8))
                    except Exception:
                        pass
                elif action.get("type") == "ltm" and ltm_manager:
                    try:
                        ltm_manager.add_memory(
                            action.get("content", ""),
                            memory_type=action.get("memory_type", "note"),
                            metadata=action.get("metadata", {"user_id": user_id}),
                            importance=action.get("importance", 0.7),
                            user_id=user_id
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        return ChatResponse(
            response=result["response"],
            reasoning=result.get("reasoning", {}),
            metadata={**result.get("metadata", {}), "user_id": user_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")






@app.post("/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Upload and process a PDF file (user-scoped).

    PDF content is extracted and stored server-side only.
    Frontend receives only a doc_id reference.
    """
    if not pdf_loader:
        raise HTTPException(status_code=503, detail="PDF loader not initialized")

    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        
        content = await file.read()
        max_upload_size = 5 * 1024 * 1024
        if len(content) > max_upload_size:
            raise HTTPException(status_code=400, detail="Uploaded PDF exceeds maximum allowed size of 5 MB")

        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        
        import uuid as _uuid
        doc_id = str(_uuid.uuid4())

        
        background_tasks.add_task(
            process_pdf_background,
            temp_file_path,
            file.filename,
            doc_id,
            user_id
        )

        return {
            "message": f"PDF '{file.filename}' upload initiated.",
            "filename": file.filename,
            "status": "processing",
            "doc_id": doc_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")


def process_pdf_background(
    file_path: str,
    filename: str,
    doc_id: str,
    user_id: str
):
    """Process PDF in background, store extracted text server-side only."""
    try:
        
        extracted = pdf_loader.extract_text(file_path)

        
        max_bytes = 200 * 1024
        if len(extracted.encode("utf-8")) > max_bytes:
            logger.error(f"Extracted text for {filename} (user {user_id}) exceeds {max_bytes} bytes; rejecting upload")
            Path(file_path).unlink(missing_ok=True)
            return

        
        try:
            from backend.redis_client import get_redis
            import os
            r = get_redis()
            key = f"pdf:{user_id}:{doc_id}"
            r.set(key, extracted)
            r.expire(key, int(os.getenv("STM_TTL", "1800")))
        except Exception:
            logger.warning(f"Could not store extracted PDF text in Redis for user {user_id}")

        logger.info(f"PDF {filename} processed for user {user_id} (doc_id={doc_id})")

        
        if pdf_loader:
            try:
                pdf_loader.load_pdf(
                    file_path,
                    metadata={"user_id": user_id, "filename": filename},
                    doc_id=doc_id,
                    user_id=user_id
                )
            finally:
                
                try:
                    from backend.redis_client import get_redis
                    r = get_redis()
                    r.delete(f"pdf:{user_id}:{doc_id}")
                except Exception:
                    pass

        
        Path(file_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Background PDF processing failed for {filename} (user {user_id}): {e}")
        Path(file_path).unlink(missing_ok=True)






@app.get("/memory/stm")
async def get_stm_memories(
    limit: int = 10,
    user_id: str = Depends(get_current_user)
):
    """Get user's recent short-term memories."""
    if not stm_manager:
        raise HTTPException(status_code=503, detail="STM manager not initialized")

    try:
        memories = stm_manager.get_all_memories(user_id)
        recent_memories = memories[-limit:]

        return {
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "importance": m.importance,
                    "metadata": m.metadata
                }
                for m in recent_memories
            ],
            "total_count": len(memories)
        }

    except Exception as e:
        logger.error(f"Error getting STM memories for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="STM retrieval failed")


@app.post("/memory/clear_stm")
async def clear_stm(user_id: str = Depends(get_current_user)):
    """Clear user's short-term memory."""
    if not stm_manager:
        raise HTTPException(status_code=503, detail="STM manager not initialized")

    try:
        stm_manager.clear_memories(user_id)
        return {"message": f"STM cleared for user {user_id}"}

    except Exception as e:
        logger.error(f"Error clearing STM for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="STM clear failed")


@app.get("/memory/ltm/search")
async def search_ltm_memories(
    query: str,
    limit: int = 10,
    user_id: str = Depends(get_current_user)
):
    """Search user's long-term memories."""
    if not ltm_manager:
        raise HTTPException(status_code=503, detail="LTM manager not initialized")

    try:
        
        results = ltm_manager.search_memories(
            query,
            limit=limit,
            user_id=user_id
        )

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error searching LTM for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="LTM search failed")


@app.get("/pdf/documents")
async def get_pdf_documents(user_id: str = Depends(get_current_user)):
    """Get user's uploaded PDF documents."""
    if not pdf_loader:
        raise HTTPException(status_code=503, detail="PDF loader not initialized")

    try:
        documents = pdf_loader.get_pdf_documents(user_id)
        return {"documents": documents, "count": len(documents)}

    except Exception as e:
        logger.error(f"Error getting PDF documents for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="PDF documents retrieval failed")


@app.delete("/pdf/{document_id}")
async def delete_pdf_document(
    document_id: str,
    user_id: str = Depends(get_current_user)
):
    """Delete user's PDF document."""
    if not pdf_loader:
        raise HTTPException(status_code=503, detail="PDF loader not initialized")

    try:
        pdf_loader.delete_pdf(document_id, user_id)
        
        try:
            from backend.redis_client import get_redis
            r = get_redis()
            r.delete(f"pdf:{user_id}:{document_id}")
        except Exception:
            pass

        return {"message": f"PDF document {document_id} deleted"}

    except Exception as e:
        logger.error(f"Error deleting PDF {document_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="PDF deletion failed")






@app.get("/system/stats")
async def system_stats(user_id: str = Depends(get_current_user)):
    """Get system statistics (user-scoped)."""
    try:
        
        pdf_count = 0
        try:
            from backend.redis_client import get_redis
            r = get_redis()
            pattern = f"pdf:{user_id}:*"
            for _ in r.scan_iter(match=pattern):
                pdf_count += 1
        except Exception:
            pdf_count = 0

        stats = {
            "health": await health_check(),
            "user_id": user_id,
            "stm_count": len(stm_manager.get_all_memories(user_id)) if stm_manager else 0,
            "pdf_count": pdf_count
        }
        return stats

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="System stats failed")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
