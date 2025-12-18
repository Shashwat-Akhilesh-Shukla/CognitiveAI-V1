# CognitiveAI – A Memory-Augmented Personal Intelligence Engine

A high-performance AI agent with short-term memory, long-term vector memory, PDF knowledge ingestion, and autonomous reasoning, built using modern LLMOps architecture.

## Core MVP Features

### 1. Short-Term Memory (STM) Manager
Maintains a rolling context window of the last N user interactions using a lightweight buffer memory with relevance scoring.

### 2. Long-Term Memory (LTM) Engine
Stores user facts, preferences, tasks, and past conversation highlights using vector embeddings.

### 3. PDF Knowledge Loader
Uses Unstructured for extraction, auto-chunks, embeds, and stores PDF content into the vector database.

### 4. Cognitive Loop (Reasoning Engine)
Implements a minimal reflection cycle: input → recall → plan → respond → update memory.

### 5. Minimal FastAPI Backend
Clean FastAPI server with endpoints:
- `/chat` - Main chat interface
- `/upload_pdf` - PDF knowledge ingestion
- `/memory_view` - Memory inspection


### 7. Frontend
Next.js interface for clean, fast deployment.

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
5. Run the frontend
   ```bash
   cd frontend
   npm run dev
   ```

## Architecture

```
Frontend (NEXT.JS) ←→ FastAPI Backend ←→ Memory System
                                      ↓
                               Pinecone Vector DB
                                      ↓
                               PDF Knowledge Base
```

## Usage

1. Upload PDFs via `/upload_pdf` to build knowledge base
2. Chat with the AI via `/chat` endpoint
3. View memory contents via `/memory_view`

The system maintains context across conversations and leverages uploaded knowledge for informed responses.
