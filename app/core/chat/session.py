"""
Chat session management with database integration.
"""

import uuid
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from langchain.memory import ConversationBufferMemory

from app.db.crud import add_chat_message, get_chat_history
from app.utils.logger import logging

class ChatSession:
    """Chat session with memory and database persistence."""

    def __init__(self, video_id: str, session_id: Optional[str] = None):
        """Initialize a chat session for a video."""
        self.video_id = video_id
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
        self._loaded_from_db = False

    def load_history_from_db(self, db: Session) -> int:
        """
        Load chat history from database.
        
        Returns:
            int: Number of messages loaded
        """
        if self._loaded_from_db:
            return len(self.memory.chat_memory.messages) // 2  # Pairs of messages
            
        history = get_chat_history(db, self.video_id, self.session_id)

        # Add messages to memory in chronological order (oldest first)
        loaded_count = 0
        for entry in history:
            self.memory.chat_memory.add_user_message(entry.message)
            self.memory.chat_memory.add_ai_message(entry.response)
            loaded_count += 1
            
        self._loaded_from_db = True
        logging.info(f"Loaded {loaded_count} message pairs from chat history")
        return loaded_count

    def save_interaction(self, db: Session, question: str, answer: str) -> bool:
        """
        Save a chat interaction to the database and memory.
        
        Returns:
            bool: Success status
        """
        try:
            add_chat_message(
                db=db,
                video_id=self.video_id,
                session_id=self.session_id,
                message=question,
                response=answer
            )
            
            # Update memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            return True
        except Exception as e:
            logging.error(f"Error saving chat interaction: {str(e)}")
            
            # Still update memory even if DB save fails
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            return False
            
    def get_memory_messages(self) -> List[Dict[str, Any]]:
        """Get memory messages in serializable format."""
        messages = []
        for msg in self.memory.chat_memory.messages:
            messages.append({
                "type": msg.type,
                "content": msg.content
            })
        return messages
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Get context information for diagnostics."""
        return {
            "video_id": self.video_id,
            "session_id": self.session_id,
            "message_count": len(self.memory.chat_memory.messages) // 2,
            "loaded_from_db": self._loaded_from_db
        }