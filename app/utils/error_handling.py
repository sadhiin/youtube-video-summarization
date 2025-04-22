"""
Centralized error handling for the application.
"""

from typing import Optional, Dict, Any
import traceback

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from app.embeddings.get_embedding_model import initalize_embedding_model
from app.config import config, chat_config
from app.utils.logger import logging

def handle_vector_store_error(error: Exception, video_id: str, transcript: str) -> Optional[FAISS]:
    """
    Handle vector store creation errors with graceful degradation.
    
    Args:
        error: The exception that occurred
        video_id: ID of the video
        transcript: Transcript text
        
    Returns:
        Optional FAISS vector store or None if all attempts fail
    """
    error_str = str(error)
    error_trace = traceback.format_exc()
    
    logging.error(f"Error creating vector store for video {video_id}: {error_str}")
    logging.error(error_trace)
    
    # Try emergency vector store creation
    try:
        logging.info("Attempting to create emergency vector store")
        
        emergency_text = transcript[:chat_config.EMERGENCY_CHUNK_LIMIT] 
        
        # Create a single document
        embeddings = initalize_embedding_model()
        document = Document(
            page_content=emergency_text,
            metadata={
                "source": "emergency_transcript", 
                "video_id": video_id
            }
        )
        
        vector_store = FAISS.from_documents([document], embeddings)
        logging.info("Successfully created emergency vector store")
        
        return vector_store
    except Exception as e:
        logging.error(f"Emergency vector store creation failed: {str(e)}")
        return None

def log_diagnostic_info(context: Dict[str, Any]):
    """
    Log diagnostic information for debugging.
    
    Args:
        context: Dictionary of diagnostic information
    """
    if not config.DEBUG:
        return
        
    try:
        import json
        logging.info(f"Diagnostic info: {json.dumps(context)}")
    except Exception as e:
        logging.error(f"Error logging diagnostic info: {str(e)}")