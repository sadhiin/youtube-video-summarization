"""
Chat handler module for interacting with video transcripts using Langchain.
"""

import os
import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from app.config import config
from app.db.crud import add_chat_message, get_chat_history, get_stored_summary
from app.db.models import Video, Transcript
from app.utils.vector_store import get_vector_store_for_video

from app.embeddings.get_embedding_model import get_embedding_model

# Chat system prompt
CHAT_SYSTEM_PROMPT = """
You are an AI assistant that helps users understand YouTube video content.
You have access to the transcript of the video they're asking about.

Answer the user's question based on the transcript provided.
Be concise and accurate. If the transcript doesn't contain the information
to answer the question, just say so instead of making up information.

Transcript context:
{context}

Chat history:
{chat_history}
"""


class ChatSession:
    """Chat session with memory."""

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

    def load_history_from_db(self, db: Session):
        """Load chat history from database."""
        history = get_chat_history(db, self.video_id, self.session_id)

        # Add messages to memory in chronological order (oldest first)
        for entry in reversed(history):
            self.memory.chat_memory.add_user_message(entry.message)
            self.memory.chat_memory.add_ai_message(entry.response)

    def save_interaction(self, db: Session, question: str, answer: str):
        """Save a chat interaction to the database."""
        add_chat_message(
            db=db,
            video_id=self.video_id,
            session_id=self.session_id,
            message=question,
            response=answer
        )
        # Also update memory
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)


def create_chat_chain(video_id: str, session: ChatSession):
    """Create a chatbot chain with chat memory for a video."""
    # Initialize LLM
    llm = init_chat_model(
        model=config.DEFAULT_SUMMARY_MODEL,
        model_provider="groq",
        temperature=0.3
    )

    # Get vector store for this video
    vector_store = get_vector_store_for_video(video_id)
    if not vector_store:
        # Fallback approach: create vector store on the fly
        from app.db.database import get_db
        db = next(get_db())
        stored_summary = get_stored_summary(db, video_id)

        if stored_summary and "transcript_text" in stored_summary and stored_summary["transcript_text"]:
            # Create embeddings for transcript text
            from app.utils.vector_store import add_to_vector_db
            print(f"Creating vector store for video {video_id}")
            add_to_vector_db(video_id, stored_summary["transcript_text"])
            vector_store = get_vector_store_for_video(video_id)
        else:
            print(f"No transcript text found for video {video_id}")

    if not vector_store:
        raise ValueError(f"No vector store available for video {video_id}")

    # Create retriever with contextual compression
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize embeddings for filtering
    embeddings = get_embedding_model()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )

    # Create chat prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=session.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain


def get_chat_response(video_id: str, message: str, db: Session, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a chat response for a question about a video.

    Args:
        video_id: YouTube video ID
        message: User's message/question
        db: Database session
        session_id: Optional session ID for continuing a conversation

    Returns:
        Dictionary with answer and sources
    """
    session = ChatSession(video_id, session_id)
    session.load_history_from_db(db)

    try:
        # Create chain
        chain = create_chat_chain(video_id, session)

        # Get response
        response = chain.invoke({
            "question": message,
            "chat_history": session.memory.chat_memory.messages
        })

        # Save interaction to database
        session.save_interaction(db, message, response["answer"])

        # Format sources
        sources = []
        if "source_documents" in response:
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        return {
            "answer": response["answer"],
            "sources": sources,
            "session_id": session.session_id
        }
    except ValueError as e:
        # Handle the case where no vector store is available (no transcript)
        error_msg = str(e)
        if "No vector store available" in error_msg:
            # Provide a more user-friendly response
            return {
                "answer": "I'm sorry, but the transcript for this video isn't available or couldn't be processed correctly. Please try summarizing the video again.",
                "sources": [],
                "session_id": session.session_id
            }
        raise