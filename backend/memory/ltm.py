"""
Long-Term Memory Manager for CognitiveAI

Uses Pinecone vector database for storing and retrieving long-term memories,
user facts, preferences, and conversation highlights with semantic search.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class LTMManager:
    """
    Long-Term Memory Manager using Pinecone vector database.

    Stores user facts, preferences, tasks, and conversation highlights
    with semantic retrieval capabilities.
    """

    def __init__(self, api_key: Optional[str] = None,
                 cloud: str = "aws",
                 region: str = "us-east-1",
                 index_name: str = "cognitiveai-ltm",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize LTM Manager.

        Args:
            api_key: Pinecone API key (from env if not provided)
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
            index_name: Name of the Pinecone index
            embedding_model: Sentence transformer model name
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.cloud = cloud
        self.region = region
        self.index_name = index_name
        self.embedding_model_name = embedding_model

        if not self.api_key:
            raise ValueError("Pinecone API key not provided. Set PINECONE_API_KEY environment variable.")

        
        self.pc = Pinecone(api_key=self.api_key)

        
        self.embedding_model = SentenceTransformer(embedding_model)

        
        self._ensure_index()

    def _ensure_index(self):
        """Ensure the Pinecone index exists with proper configuration."""
        try:
            
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                
                vector_size = self.embedding_model.get_sentence_embedding_dimension()

                self.pc.create_index(
                    name=self.index_name,
                    dimension=vector_size,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")

            
            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise

    def add_memory(self, content: str, memory_type: str = "general",
                   metadata: Optional[Dict[str, Any]] = None,
                   importance: float = 1.0,
                   user_id: Optional[str] = None) -> str:
        """
        Add a memory to long-term storage.

        Args:
            content: The memory content
            memory_type: Type of memory (general, fact, preference, task, etc.)
            metadata: Additional metadata
            importance: Importance score (0.0-1.0)

        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())

        
        embedding = self.embedding_model.encode(content).tolist()

        
        metadata_dict = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "timestamp": self._get_timestamp(),
            "id": memory_id
        }

        if metadata:
            metadata_dict.update(metadata)

        
        if user_id:
            metadata_dict.setdefault("user_id", user_id)

        
        self.index.upsert(vectors=[(memory_id, embedding, metadata_dict)])

        logger.info(f"Added LTM memory: {memory_id} (type: {memory_type})")
        return memory_id

    def search_memories(self, query: str, memory_type: Optional[str] = None,
                       limit: int = 10, score_threshold: float = 0.0,
                       user_id: Optional[str] = None, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using semantic similarity.

        Args:
            query: Search query
            memory_type: Filter by memory type (optional)
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of memory results with scores
        """
        
        query_embedding = self.embedding_model.encode(query).tolist()

        
        filter_dict = {}
        if memory_type:
            filter_dict["memory_type"] = {"$eq": memory_type}
        if user_id:
            filter_dict["user_id"] = {"$eq": user_id}
        
        if metadata_filters:
            for k, v in metadata_filters.items():
                filter_dict[k] = {"$eq": v}

        
        search_response = self.index.query(
            vector=query_embedding,
            filter=filter_dict if filter_dict else None,
            top_k=limit,
            include_metadata=True,
            include_values=False
        )

        
        results = []
        for match in search_response.matches:
            if match.score >= score_threshold:
                metadata = match.metadata or {}
                result = {
                    "id": match.id,
                    "content": metadata.get("content", ""),
                    "memory_type": metadata.get("memory_type", ""),
                    "importance": metadata.get("importance", 1.0),
                    "timestamp": metadata.get("timestamp", 0),
                    "score": match.score,
                    "metadata": {k: v for k, v in metadata.items()
                               if k not in ["content", "memory_type", "importance", "timestamp", "id"]}
                }
                results.append(result)

        return results

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory data or None if not found
        """
        try:
            
            result = self.index.query(
                vector=[0.0] * self.embedding_model.get_sentence_embedding_dimension(),  
                filter={"id": {"$eq": memory_id}},
                top_k=1,
                include_metadata=True,
                include_values=False
            )

            if result.matches:
                match = result.matches[0]
                metadata = match.metadata or {}
                return {
                    "id": memory_id,
                    "content": metadata.get("content", ""),
                    "memory_type": metadata.get("memory_type", ""),
                    "importance": metadata.get("importance", 1.0),
                    "timestamp": metadata.get("timestamp", 0),
                    "metadata": {k: v for k, v in metadata.items()
                               if k not in ["content", "memory_type", "importance", "timestamp", "id"]}
                }
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")

        return None

    def update_memory(self, memory_id: str, content: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     importance: Optional[float] = None):
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            metadata: New metadata (optional)
            importance: New importance score (optional)
        """
        
        current = self.get_memory_by_id(memory_id)
        if not current:
            raise ValueError(f"Memory {memory_id} not found")

        
        new_metadata = dict(current)
        new_metadata.pop('id', None)  

        if content is not None:
            new_metadata["content"] = content

        if importance is not None:
            new_metadata["importance"] = importance

        if metadata:
            new_metadata.update(metadata)

        
        new_metadata["timestamp"] = self._get_timestamp()

        
        embedding = self.embedding_model.encode(content if content is not None else current["content"]).tolist()

        
        self.index.delete(ids=[memory_id])
        self.index.upsert(vectors=[(memory_id, embedding, new_metadata)])

        logger.info(f"Updated LTM memory: {memory_id}")

    def delete_memory(self, memory_id: str):
        """
        Delete a memory from long-term storage.

        Args:
            memory_id: Memory ID to delete
        """
        self.index.delete(ids=[memory_id])
        logger.info(f"Deleted LTM memory: {memory_id}")

    def delete_memories_by_user(self, user_id: str, batch_size: int = 1000):
        """Delete all LTM memories for a given user by metadata filter.

        Note: This may be slow for very large indexes; it uses a single query
        with a large `top_k` and deletes returned ids.
        """
        if not user_id:
            raise ValueError("user_id is required")

        try:
            filter_dict = {"user_id": {"$eq": user_id}}
            
            resp = self.index.query(
                vector=[0.0] * self.embedding_model.get_sentence_embedding_dimension(),
                filter=filter_dict,
                top_k=batch_size,
                include_metadata=True,
                include_values=False
            )

            ids = [m.id for m in getattr(resp, "matches", [])]
            if ids:
                self.index.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} LTM memories for user {user_id}")
            else:
                logger.info(f"No LTM memories found for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to delete LTM memories for user {user_id}: {e}")
            raise

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the long-term memory index."""
        try:
            
            index_stats = self.index.describe_index_stats()
            return {
                "total_memories": index_stats.total_vector_count,
                "index_name": self.index_name,
                "dimension": index_stats.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    def clear_all_memories(self):
        """Clear all memories from the index (dangerous operation)."""
        try:
            
            self.pc.delete_index(self.index_name)
            self._ensure_index()
            logger.warning("Cleared all LTM memories")
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            raise

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()

    def add_user_profile(self, profile_data: Dict[str, Any]):
        """
        Add user profile information to memory.

        Args:
            profile_data: User profile data (name, preferences, etc.)
        """
        content = f"User Profile: {profile_data}"
        self.add_memory(
            content=content,
            memory_type="user_profile",
            metadata=profile_data,
            importance=0.9  
        )

    def add_conversation_highlight(self, conversation_text: str,
                                 highlight_type: str = "important",
                                 user_id: Optional[str] = None):
        """
        Add a conversation highlight to memory.

        Args:
            conversation_text: The highlighted conversation text
            highlight_type: Type of highlight (important, decision, etc.)
        """
        content = f"Conversation Highlight ({highlight_type}): {conversation_text}"
        metadata = {"highlight_type": highlight_type}
        if user_id:
            metadata["user_id"] = user_id
        self.add_memory(
            content=content,
            memory_type="conversation_highlight",
            metadata=metadata,
            importance=0.8,
            user_id=user_id
        )

    def add_task(self, task_description: str, status: str = "pending",
                priority: str = "medium"):
        """
        Add a task to memory.

        Args:
            task_description: Description of the task
            status: Task status (pending, in_progress, completed)
            priority: Task priority (low, medium, high)
        """
        content = f"Task ({status}, {priority}): {task_description}"
        self.add_memory(
            content=content,
            memory_type="task",
            metadata={
                "status": status,
                "priority": priority,
                "task_description": task_description
            },
            importance=0.7
        )
