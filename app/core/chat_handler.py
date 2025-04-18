# -*- coding: utf-8 -*-
"""
Chat handler module for interacting with video transcripts using Langchain.
"""

import os
import traceback
import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document

from app.config import config
from app.db.crud import add_chat_message, get_chat_history, get_stored_summary
from app.db.models import Video, Transcript
from app.utils.vector_store import get_vector_store_for_video
from app.embeddings.get_embedding_model import initalize_embedding_model
from app.utils.logger import logging

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
    logging.info(f"Checking vector store for video {video_id}: {'Found' if vector_store else 'Not found'}")

    # If no vector store, get transcript and create one
    if not vector_store:
        from app.db.database import get_db
        db = next(get_db())
        stored_summary = get_stored_summary(db, video_id)   # this will return the summary and transcript and video info

        if not stored_summary or not stored_summary.get("transcript_text"):
            raise ValueError(f"No transcript available for video {video_id}")

        transcript_text = stored_summary["transcript_text"]
        logging.info(f"Creating vector store with transcript of length: {len(transcript_text)}")

        # Try to create vector store
        from app.utils.vector_store import add_to_vector_db
        vector_store = add_to_vector_db(video_id, transcript_text)

        if not vector_store:
            # Emergency fallback: create minimal vector store directly here
            logging.info("Creating emergency vector store...")
            from langchain_community.vectorstores import FAISS
            from langchain.docstore.document import Document

            # Split text into smaller chunks for emergency storage
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = []
            for i, chunk in enumerate(text_splitter.split_text(transcript_text)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": "transcript", "chunk_id": i, "video_id": video_id}
                ))

            if not docs:
                # Last resort: use whole transcript as one chunk
                docs = [Document(
                    page_content=transcript_text[:9000],  # Limit to first 9000 chars
                    metadata={"source": "emergency_transcript", "video_id": video_id}
                )]

            # Create vector store directly
            embeddings = initalize_embedding_model()
            vector_store = FAISS.from_documents(docs, embeddings)
            logging.info(f"Created emergency vector store with {len(docs)} documents")

    if not vector_store:
        raise ValueError("Failed to create vector store for this video")

    # Test retriever functionality
    test_retrieval = vector_store.similarity_search("test", k=1)
    logging.info(f"Test retrieval returned {len(test_retrieval)} documents")
    if test_retrieval:
        logging.info(f"Sample content length: {len(test_retrieval[0].page_content)}")

    # Create retriever with better search parameters
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,  # Return 5 most relevant chunks
            "fetch_k": 10  # Consider more chunks before filtering
        }
    )

    # Create contextual compression for better filtering
    embeddings = initalize_embedding_model()
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.5
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )

    # Create improved system prompt template
    system_template = """
    You are an AI assistant that helps users understand YouTube video content.
    You have access to the transcript of the video they're asking about.

    Below is the relevant context from the video transcript:

    {context}

    Previous conversation history:
    {chat_history}

    Based ONLY on the information provided in the transcript context above,
    answer the user's question thoroughly and accurately.

    If the transcript doesn't contain information to answer the question,
    be honest and say you don't have that information from the video.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])

    # Create conversational chain with explicit parameters
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=session.memory,
        return_source_documents=True,
        verbose=True,  # Enable verbose mode for debugging
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )
    
    # TODO: Need to validate the chain and retriever chain creation.

    return chain


def get_chat_response(video_id: str, message: str, db: Session, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a chat response for a question about a video.
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
    assert stored_summary.get("transcript_text") is not None, "Transcript text should not be None"
    
    if not stored_summary.get("transcript_text"):
        return {
            "answer": "The transcript for this video seems to be missing. Please try summarizing the video again.",
            "sources": [],
            "session_id": session.session_id
        }

    # logging.info debug info
    transcript_length = len(stored_summary.get("transcript_text", ""))
    logging.info(f"Found transcript for video {video_id} with transcription text length: {transcript_length}")

    try:
        # Create chain
        chain = create_chat_chain(video_id, session)

        # Test retrieval directly
        vector_store = get_vector_store_for_video(video_id)
        if vector_store:
            test_docs = vector_store.similarity_search(message, k=3)
            if not test_docs:
                logging.warning("Vector store has no relevant chunks for test query!")
            else:
                logging.info(f"Direct search found {len(test_docs)} relevant chunks")
                logging.info(f"Sample chunk: {test_docs[0].page_content[:100] if test_docs else 'None'}")

        # Get response
        logging.info(f"Processing question: {message}")
        response = chain.invoke({
            "question": message,
            "chat_history": session.memory.chat_memory.messages
        })

       
        if "source_documents" in response and response["source_documents"]:
            logging.info(f"Retrieved {len(response['source_documents'])} source documents")
        else:
            llm = init_chat_model(
                model=config.DEFAULT_SUMMARY_MODEL,
                model_provider="groq",
                temperature=0.3
            )

            logging.warning("WARNING: No source documents were retrieved!")
            # Fallback to direct transcript usage
            transcript_excerpt = stored_summary["transcript_text"]
            direct_prompt = ChatPromptTemplate.from_messages([
                ("system", f"Answer based on this transcript start: {transcript_excerpt}..."),
                ("human", "{question}")
            ])
            response = direct_prompt | llm
            answer = response.invoke({"question": message}).content
            session.save_interaction(db, message, answer)
            # Format sources
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    sources.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })

            return {
                "answer": answer,
                'sources': [],
                "session_id": session.session_id
            }

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
        # Handle the case where no vector store is available
        logging.error(f"Error in chat response: {e.__str__()}")

        # Try direct approach using the complete transcript
        try:
            llm = init_chat_model(
                model=config.DEFAULT_SUMMARY_MODEL,
                model_provider="groq",
                temperature=0.3
            )

            transcript_text = stored_summary["transcript_text"]

            direct_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an AI assistant that helps users understand YouTube video content.
                Here's a part of the video transcript (the beginning of the video):

                {transcript_text}

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
            logging.error(f"Direct approach also failed: {direct_error}")
            return {
                "answer": "I'm having trouble processing the transcript. Please try again or summarize the video again.",
                "sources": [],
                "session_id": session.session_id
            }
    except Exception as e:
        logging.error(f"Unexpected error in chat_handler: {str(e)}")
        logging.error(traceback.format_exc())

        return {
            "answer": "I encountered an unexpected error while processing your question. Please try again later.",
            "sources": [],
            "session_id": session.session_id
        }