"""
FastAPI Backend for CognitiveAI

Provides REST API endpoints for chat, PDF upload, and memory inspection.
"""

import os
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from dotenv import load_dotenv
load_dotenv("backend/.env")
from backend.memory.stm import STMManager
from backend.memory.ltm import LTMManager
from .pdf_loader import PDFLoader
from .reasoning import CognitiveReasoningEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="CognitiveAI API",
    description="Memory-Augmented Personal Intelligence Engine",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


stm_manager = None
ltm_manager = None
pdf_loader = None
reasoning_engine = None
pdf_text_store: Dict[str, str] = {}


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    user_id: Optional[str] = "default"
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


def initialize_memory_systems():
    """Initialize all memory systems and reasoning engine."""
    global stm_manager, ltm_manager, pdf_loader, reasoning_engine

    try:
        
        stm_manager = STMManager(max_size=50, decay_factor=0.95)

        
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
        pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
        if not pinecone_api_key:
            logger.warning("PINECONE_API_KEY not set — LTM disabled for local/dev.")
            ltm_manager = None
        else:
            ltm_manager = LTMManager(
                api_key=pinecone_api_key,
                cloud=pinecone_cloud,
                region=pinecone_region,
                index_name="cognitiveai-ltm"
            )

        
        pdf_loader = PDFLoader(ltm_manager)

        
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if not perplexity_api_key:
            logger.warning("PERPLEXITY_API_KEY not set. Chat functionality will not work.")

        reasoning_engine = CognitiveReasoningEngine(
            stm_manager=stm_manager,
            ltm_manager=ltm_manager,
            pdf_loader=pdf_loader,
            perplexity_api_key=perplexity_api_key or ""
        )

        logger.info("Memory systems initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize memory systems: {e}")
        
        logger.warning("Continuing without full memory system initialization.")


@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup."""
    initialize_memory_systems()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "CognitiveAI API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "systems": {
            "stm": stm_manager is not None,
            "ltm": ltm_manager is not None,
            "pdf_loader": pdf_loader is not None,
            "reasoning_engine": reasoning_engine is not None
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Processes user messages through the cognitive reasoning engine.
    """
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")

    try:
        
        full_message = request.message
        if getattr(request, 'doc_id', None):
            extracted = pdf_text_store.get(request.doc_id)
            if extracted:
                full_message = f"{full_message}\n\nextracted pdf text: \"{extracted}\""

        result = reasoning_engine.process_message(
            user_message=full_message,
            user_id=request.user_id or "default"
        )

        return ChatResponse(
            response=result["response"],
            reasoning=result["reasoning"],
            metadata=result["metadata"]
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """
    Upload and process a PDF file.

    The PDF will be processed in the background and its content stored in the knowledge base.
    """
    if not pdf_loader:
        raise HTTPException(status_code=503, detail="PDF loader not initialized")

    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        
        pdf_metadata = {}
        if metadata:
            try:
                import json
                pdf_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
            except ImportError:
                raise HTTPException(status_code=500, detail="JSON parsing not available")

        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        
        import uuid as _uuid
        doc_id = str(_uuid.uuid4())

        
        background_tasks.add_task(
            process_pdf_background,
            temp_file_path,
            file.filename,
            pdf_metadata,
            doc_id
        )

        return {
            "message": f"PDF '{file.filename}' upload initiated. Processing in background.",
            "filename": file.filename,
            "status": "processing",
            "doc_id": doc_id
        }

    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")


def process_pdf_background(file_path: str, filename: str, metadata: Dict[str, Any], doc_id: str):
    """Process PDF in background task."""
    try:
        
        try:
            extracted = pdf_loader.extract_text(file_path)
            pdf_text_store[doc_id] = extracted
            logger.info(f"Extracted text for {filename} stored under doc_id={doc_id}")
        except Exception as e:
            logger.error(f"Failed to extract text synchronously for {filename}: {e}")

        
        stored_doc_id = pdf_loader.load_pdf(file_path, metadata, doc_id=doc_id)
        logger.info(f"Successfully processed PDF: {filename} (ID: {stored_doc_id})")

        
        Path(file_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Background PDF processing failed for {filename}: {e}")
        
        Path(file_path).unlink(missing_ok=True)


@app.get("/memory_view", response_model=MemoryStats)
async def memory_view():
    """
    Get memory statistics and recent activity.

    Returns overview of STM, LTM, and PDF knowledge base status.
    """
    if not all([stm_manager, ltm_manager, pdf_loader, reasoning_engine]):
        raise HTTPException(status_code=503, detail="Memory systems not fully initialized")

    try:
        
        stm_memories = stm_manager.get_all_memories()
        stm_count = len(stm_memories)

        
        ltm_stats = ltm_manager.get_memory_stats()

        
        pdf_documents = pdf_loader.get_pdf_documents()

        
        reasoning_stats = reasoning_engine.get_reasoning_stats()

        return MemoryStats(
            stm_count=stm_count,
            ltm_stats=ltm_stats,
            pdf_documents=pdf_documents,
            reasoning_stats=reasoning_stats
        )

    except Exception as e:
        logger.error(f"Error getting memory view: {e}")
        raise HTTPException(status_code=500, detail=f"Memory view failed: {str(e)}")


@app.get("/memory/stm")
async def get_stm_memories(limit: int = 10):
    """Get recent short-term memories."""
    if not stm_manager:
        raise HTTPException(status_code=503, detail="STM manager not initialized")

    try:
        memories = stm_manager.get_all_memories()
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
        logger.error(f"Error getting STM memories: {e}")
        raise HTTPException(status_code=500, detail=f"STM retrieval failed: {str(e)}")


@app.get("/memory/ltm/search")
async def search_ltm_memories(query: str, limit: int = 10):
    """Search long-term memories."""
    if not ltm_manager:
        raise HTTPException(status_code=503, detail="LTM manager not initialized")

    try:
        results = ltm_manager.search_memories(query, limit=limit)

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error searching LTM: {e}")
        raise HTTPException(status_code=500, detail=f"LTM search failed: {str(e)}")


@app.get("/pdf/documents")
async def get_pdf_documents():
    """Get list of uploaded PDF documents."""
    if not pdf_loader:
        raise HTTPException(status_code=503, detail="PDF loader not initialized")

    try:
        documents = pdf_loader.get_pdf_documents()
        return {"documents": documents, "count": len(documents)}

    except Exception as e:
        logger.error(f"Error getting PDF documents: {e}")
        raise HTTPException(status_code=500, detail=f"PDF documents retrieval failed: {str(e)}")


@app.delete("/pdf/{document_id}")
async def delete_pdf_document(document_id: str):
    """Delete a PDF document and its knowledge."""
    if not pdf_loader:
        raise HTTPException(status_code=503, detail="PDF loader not initialized")

    try:
        pdf_loader.delete_pdf(document_id)
        return {"message": f"PDF document {document_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting PDF {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"PDF deletion failed: {str(e)}")


@app.post("/memory/clear_stm")
async def clear_stm():
    """Clear short-term memory."""
    if not stm_manager:
        raise HTTPException(status_code=503, detail="STM manager not initialized")

    try:
        stm_manager.clear_memories()
        return {"message": "Short-term memory cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing STM: {e}")
        raise HTTPException(status_code=500, detail=f"STM clear failed: {str(e)}")


@app.get("/system/stats")
async def system_stats():
    """Get overall system statistics."""
    try:
        stats = {
            "health": await health_check(),
            "memory": await memory_view() if all([stm_manager, ltm_manager, pdf_loader, reasoning_engine]) else None
        }
        return stats

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"System stats failed: {str(e)}")


if __name__ == "__main__":
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
