# -*- coding: utf-8 -*-
"""
Chat handler module for interacting with video transcripts using Langchain.
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter


from app.config import config, ChatConfig
from app.db.crud import add_chat_message, get_chat_history, get_stored_summary
from app.db.models import Video, Transcript
from app.utils.vector_store import get_vector_store_for_video
from app.utils.vector_store_manager import VectorStoreManager
from app.embeddings.get_embedding_model import initalize_embedding_model
from app.utils.logger import logging
from app.core.prompts import system_template
from langchain.memory import ConversationBufferMemory
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


# from typing import Optional
# from uuid import uuid4
# from sqlalchemy.orm import Session
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import StateGraph, MessagesState

# class ChatSession:
#     """Modern chat session using LangGraph memory management."""
    
#     def __init__(
#         self,
#         video_id: str,
#         session_id: Optional[str] = None,
#         window_size: int = 5
#     ):
#         self.video_id = video_id
#         self.session_id = session_id or str(uuid4())
#         self.checkpointer = MemorySaver()
#         self.workflow = self._create_workflow(window_size)

#     def _create_workflow(self, window_size: int) -> StateGraph:
#         """Configure LangGraph workflow with memory trimming."""
#         workflow = StateGraph(state_schema=MessagesState)

#         # Define model invocation with context window management
#         def model_invoke(state: MessagesState):
#             from langchain_core.messages import trim_messages
        
#             trimmed = trim_messages(
#                 state.messages,
#                 max_tokens=window_size,
#                 token_counter=lambda m: 1  # Count each message as 1 "token"
#             )
#             return {"messages": trimmed}

#         workflow.add_node("model", model_invoke)
#         workflow.set_entry_point("model")
#         return workflow.compile(checkpointer=self.checkpointer)

#     def process_message(self, user_input: str) -> str:
#         """Process user message with persistent memory."""
#         config = {"configurable": {"thread_id": self.session_id}}
        
#         # Invoke workflow with message history
#         result = self.workflow.invoke(
#             {"messages": [HumanMessage(content=user_input)]},
#             config=config
#         )
        
#         # Extract and return AI response
#         return next(
#             msg.content 
#             for msg in result["messages"] 
#             if isinstance(msg, AIMessage)
#         )

#     def get_history(self) -> list:
#         """Retrieve conversation history from checkpointer."""
#         config = {"configurable": {"thread_id": self.session_id}}
#         return self.checkpointer.get(config)["messages"]

#     ## Database integration remains similar but uses checkpointer state
# def load_history(db: Session, session: ChatSession):
#     """Sync database history with LangGraph checkpointer"""
#     db_history = get_chat_history(db, session.video_id, session.session_id)
#     messages = [
#         HumanMessage(content=entry.message) if entry.is_user 
#         else AIMessage(content=entry.response)
#         for entry in db_history
#     ]
    
#     # Initialize checkpointer state
#     session.checkpointer.put(
#         {"configurable": {"thread_id": session.session_id}},
#         {"messages": messages}
#     )


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
        similarity_threshold=0.2
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )

    # Create improved system prompt template
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])
    logging.info(f"*********** Calling the ConversationalRetrievalChain with video_id: {video_id} ***********")
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


def get_chat_response(video_id: str, message: str, db: Session, session:ChatSession) -> Dict[str, Any]:
    """
    Get a chat response for a question about a video.
    """
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