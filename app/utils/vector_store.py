"""
Vector store operations using FAISS for transcript embeddings.
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

# from langchain.embeddings import init_embeddings
from app.embeddings.get_embedding_model import get_embedding_model
# from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import config

# Global storage for vector stores
_VIDEO_VECTOR_STORES = {}

# Directory for storing FAISS indices
VECTOR_DIR = config.BASE_DIR / "data" / "vector_indices"


def init_vector_store():
    """Initialize the vector store directory."""
    os.makedirs(VECTOR_DIR, exist_ok=True)
    # Load existing indices
    _load_existing_indices()


def _load_existing_indices():
    """Load existing vector indices from disk."""
    indices_dir = Path(VECTOR_DIR)
    if not indices_dir.exists():
        return

    # Get embeddings
    embeddings = get_embedding_model()

    # Load each index
    for video_dir in indices_dir.iterdir():
        if video_dir.is_dir():
            video_id = video_dir.name
            index_path = video_dir / "index"
            if index_path.exists():
                try:
                    vector_store = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
                    _VIDEO_VECTOR_STORES[video_id] = vector_store
                    print(f"Loaded vector index for video {video_id}")
                except Exception as e:
                    print(f"Error loading vector index for video {video_id}: {e}")


def get_vector_store_for_video(video_id: str) -> Optional[FAISS]:
    """Get the vector store for a specific video."""
    return _VIDEO_VECTOR_STORES.get(video_id)


def add_to_vector_db(video_id: str, text: str):
    """
    Add a transcript to the vector database.
    """
    if not text:
        print(f"Error: Transcript for video {video_id} is empty")
        return None

    # Strip and check the text length
    cleaned_text = text.strip()
    if len(cleaned_text) < 10:
        print(f"Error: Transcript for video {video_id} is too short ({len(cleaned_text)} chars)")
        return None

    # Create the folder for this video's index
    video_dir = VECTOR_DIR / video_id
    os.makedirs(video_dir, exist_ok=True)

    # Split text into chunks - INCREASED CHUNK SIZE FOR BETTER CONTEXT
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased from 500
        chunk_overlap=200,  # Increased from 50
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Create documents with metadata
    chunks = text_splitter.split_text(cleaned_text)
    print(f"Split transcript into {len(chunks)} chunks for vector storage")

    # Ensure we have chunks to work with
    if not chunks:
        print(f"Error: No chunks were created from transcript for video {video_id}")
        return None

    # Store original text in metadata for debugging
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "chunk_id": i,
                "source": "transcript",
                "chunk_length": len(chunk)  # Add length metadata for debugging
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    # Save metadata about the chunks for debugging
    try:
        with open(video_dir / "chunks_metadata.json", "w") as f:
            json.dump({
                "num_chunks": len(chunks),
                "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks),
                "total_length": len(cleaned_text)
            }, f)
    except Exception as e:
        print(f"Warning: Could not save chunks metadata: {e}")

    # Initialize embeddings
    embeddings = get_embedding_model()

    try:
        # Create vector store
        vector_store = FAISS.from_documents(documents, embeddings)

        # Test the vector store with a simple query to ensure it works
        test_results = vector_store.similarity_search("test query", k=1)
        if not test_results:
            print("Warning: Vector store created but test query returned no results")

        # Save the vector store
        index_path = video_dir / "index"
        vector_store.save_local(str(index_path))

        # Add to global registry
        _VIDEO_VECTOR_STORES[video_id] = vector_store

        print(f"Successfully added {len(documents)} chunks to vector store for video {video_id}")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store for video {video_id}: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def search_similar_videos(query: str, limit: int = 5) -> List[str]:
    """
    Search for videos with similar content based on the query.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        List of video IDs
    """
    if not _VIDEO_VECTOR_STORES:
        return []

    # Initialize embeddings
    embeddings = get_embedding_model()

    # Convert query to embedding
    query_embedding = embeddings.embed_query(query)

    # Search across all vector stores
    results = []
    for video_id, vector_store in _VIDEO_VECTOR_STORES.items():
        try:
            # Search in this vector store
            similar_docs = vector_store.similarity_search_by_vector(
                query_embedding,
                k=2  # Get top 2 chunks from each video
            )

            if similar_docs:
                # Calculate average similarity
                # (This is simplified - in a real app you might use a more sophisticated approach)
                results.append({
                    "video_id": video_id,
                    "similarity": 1.0,  # Placeholder for actual similarity score
                    "matching_chunks": len(similar_docs)
                })
        except Exception as e:
            print(f"Error searching vector store for video {video_id}: {e}")

    # Sort by relevance and return video IDs
    results.sort(key=lambda x: (x["matching_chunks"], x["similarity"]), reverse=True)
    return [r["video_id"] for r in results[:limit]]


def delete_vector_store(video_id: str):
    """Delete a vector store for a video."""
    # Remove from memory
    if video_id in _VIDEO_VECTOR_STORES:
        del _VIDEO_VECTOR_STORES[video_id]

    # Remove from disk
    video_dir = VECTOR_DIR / video_id
    if video_dir.exists():
        import shutil
        shutil.rmtree(video_dir)