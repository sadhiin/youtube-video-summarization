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
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
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
    print(f"Checking vector store for video {video_id}: {'Found' if vector_store else 'Not found'}")

    if not vector_store:
        # Fallback approach: create vector store on the fly
        from app.db.database import get_db
        db = next(get_db())
        stored_summary = get_stored_summary(db, video_id)

        if not stored_summary:
            print(f"No stored summary found for video {video_id}")
            raise ValueError(f"No stored summary available for video {video_id}")

        if not stored_summary.get("transcript_text"):
            print(f"No transcript text found in stored summary for video {video_id}")
            raise ValueError(f"No transcript text available for video {video_id}")

        # Create embeddings for transcript text
        from app.utils.vector_store import add_to_vector_db
        print(f"Creating vector store for video {video_id}, transcript length: {len(stored_summary['transcript_text'])}")
        vector_store = add_to_vector_db(video_id, stored_summary["transcript_text"])

        if not vector_store:
            print(f"Failed to create vector store for video {video_id}")
            # Try one more approach - create a very simple vector store with minimal content
            try:
                # Create a simple document with the first 1000 characters of transcript
                doc = Document(
                    page_content=stored_summary["transcript_text"][:1000],
                    metadata={"video_id": video_id, "chunk_id": 0, "source": "transcript"}
                )

                # Initialize embeddings
                embeddings = get_embedding_model()

                # Create vector store with a single document
                vector_store = FAISS.from_documents([doc], embeddings)

                # Add to global registry
                from app.utils.vector_store import _VIDEO_VECTOR_STORES
                _VIDEO_VECTOR_STORES[video_id] = vector_store
                print(f"Created emergency fallback vector store for video {video_id}")
            except Exception as e:
                print(f"Failed to create emergency vector store: {e}")
                import traceback
                print(traceback.format_exc())

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

    # Create chat prompt - properly formatted for ConversationalRetrievalChain
    system_template = """
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])

    # Create conversational chain with explicit document variable mapping
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=session.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context" 
        }
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

    # First check if the transcript exists in the database
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

    # Print debug info
    transcript_length = len(stored_summary.get("transcript_text", ""))
    print(f"Found transcript for video {video_id} with length: {transcript_length}")

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
        print(f"Error in chat response: {error_msg}")

        if "No vector store available" in error_msg:
            # Try a direct approach without vector store
            try:
                # If we get here, we have the transcript but couldn't create a vector store
                # Let's try to use the LLM directly with the first part of the transcript
                llm = init_chat_model(
                    model=config.DEFAULT_SUMMARY_MODEL,
                    model_provider="groq",
                    temperature=0.3
                )

                # Create a simple prompt with the first part of the transcript
                transcript_excerpt = stored_summary["transcript_text"][:2000]  # First 2000 chars

                direct_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are an AI assistant that helps users understand YouTube video content.
                    Here's a part of the video transcript:

                    {transcript_excerpt}

                    Answer the user's question based on this excerpt. If you can't answer from this excerpt,
                    say that you have limited information from the transcript."""),
                    ("human", "{question}")
                ])

                chain = direct_prompt | llm
                response = chain.invoke({"question": message})

                session.save_interaction(db, message, response.content)

                return {
                    "answer": response.content,
                    "sources": [],
                    "session_id": session.session_id
                }
            except Exception as direct_error:
                print(f"Direct approach also failed: {direct_error}")
                # Fallback to user-friendly response
                return {
                    "answer": "I'm sorry, but I'm having trouble processing the transcript for this video. Please try summarizing the video again.",
                    "sources": [],
                    "session_id": session.session_id
                }
        # For other errors
        return {
            "answer": f"I encountered an error while trying to answer your question: {error_msg}",
            "sources": [],
            "session_id": session.session_id
        }
    except Exception as e:
        print(f"Unexpected error in chat_handler: {str(e)}")
        import traceback
        print(traceback.format_exc())

        return {
            "answer": "I encountered an unexpected error while processing your question. Please try again later.",
            "sources": [],
            "session_id": session.session_id
        }