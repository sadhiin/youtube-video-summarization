# app/chat/handler.py
"""
Chat handler module for interacting with video transcripts using Langchain.
"""

import traceback
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from app.config import config
from app.db.crud import get_stored_summary
from app.utils.logger import logging
from app.core.chat.session import ChatSession
from app.core.chat.chain import ChatChainFactory
from app.core.vectorstore.manager import VectorStoreManager

class ChatHandler:
    """Handler for chat operations with video transcripts."""
    
    @staticmethod
    def get_chat_response(
        video_id: str, 
        message: str, 
        db: Session,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a chat response for a question about a video.
        
        Args:
            video_id: ID of the video
            message: User message/question
            db: Database session
            session_id: Optional session ID
            
        Returns:
            Dict with response data including answer and sources
        """
        # Input validation
        if not message or not message.strip():
            return {
                "answer": "Please provide a question about the video.",
                "sources": [],
                "session_id": session_id or "new_session"
            }
        
        session = ChatSession(video_id, session_id)
        session.load_history_from_db(db)
        
        stored_summary = get_stored_summary(db, video_id)
        if not stored_summary:
            return {
                "answer": "I couldn't find any information about this video. Please make sure the video has been summarized first.",
                "sources": [],
                "session_id": session.session_id
            }
        
        if not stored_summary.get("transcript_text"):
            return {
                "answer": "The transcript for this video seems to be missing. Please try summarizing the video again.",
                "sources": [],
                "session_id": session.session_id
            }
        
        # Diagnostic information
        transcript_length = len(stored_summary.get("transcript_text", ""))
        logging.info(f"Found transcript for video {video_id} with length: {transcript_length}")
        
        # Process the question
        try:
            return ChatHandler._process_question(video_id, message, session, stored_summary, db)
        except Exception as e:
            logging.error(f"Error in chat handler: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Try fallback approach
            return ChatHandler._try_fallback_approach(video_id, message, session, stored_summary, db)
            
    @staticmethod
    def _process_question(
        video_id: str, 
        message: str, 
        session: ChatSession, 
        summary: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Process a question using the main approach."""
        # Create chain
        chain = ChatChainFactory.create_chain(video_id, session, summary)
        
        # Test retrieval for diagnostics
        if config.DEBUG:
            vector_store = VectorStoreManager.get_vector_store(video_id)
            if vector_store:
                test_docs = vector_store.similarity_search(message, k=2)
                if test_docs:
                    logging.info(f"Direct search found {len(test_docs)} relevant chunks")
                    logging.info(f"Sample chunk: {test_docs[0].page_content[:100]}")
                else:
                    logging.warning("Vector store has no relevant chunks for test query")
        
        # Get response
        logging.info(f"Processing question: {message}")
        response = chain.invoke({
            "question": message,
            "chat_history": session.memory.chat_memory.messages
        })
        
        # Save interaction
        session.save_interaction(db, message, response["answer"])
        
        # Format sources
        sources = []
        if "source_documents" in response and response["source_documents"]:
            logging.info(f"Retrieved {len(response['source_documents'])} source documents")
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        else:
            logging.warning("No source documents were retrieved")
        
        return {
            "answer": response["answer"],
            "sources": sources,
            "session_id": session.session_id
        }
        
    @staticmethod
    def _try_fallback_approach(
        video_id: str, 
        message: str, 
        session: ChatSession, 
        summary: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Try a fallback approach when the main approach fails."""
        logging.info("Using fallback approach with direct transcript access")
        
        try:
            # Get transcript text
            transcript_text = summary["transcript_text"]
            
            # Create fallback chain
            chain = ChatChainFactory.create_fallback_chain(message, transcript_text)
            
            # Get response
            response = chain.invoke({"question": message})
            answer = response.content
            
            # Save interaction
            session.save_interaction(db, message, answer)
            
            return {
                "answer": answer,
                "sources": [],  # No sources in fallback mode
                "session_id": session.session_id,
                "fallback_used": True
            }
        except Exception as fallback_error:
            logging.error(f"Fallback approach also failed: {fallback_error}")
            
            # Ultimate fallback
            return {
                "answer": "I'm having trouble processing this video's transcript. Please try again later or summarize the video again.",
                "sources": [],
                "session_id": session.session_id,
                "error": True
            }