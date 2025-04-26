# app/chat/chain.py
"""
Chain creation and configuration for chat with video transcripts.
"""

from typing import Optional, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from app.config import config
from app.utils.logger import logging
from app.core.chat.session import ChatSession
from app.core.vectorstore.manager import VectorStoreManager
from app.embeddings.get_embedding_model import initalize_embedding_model
from app.core.prompts import CHAT_SYSTEM_TEMPLATE

class ChatChainFactory:
    """Factory for creating chat chains with appropriate configuration."""
    
    @staticmethod
    def create_chain(video_id: str, session: ChatSession, summary: Optional[Dict[str, Any]] = None) -> ConversationalRetrievalChain:
        """
        Create a chatbot chain with chat memory for a video.
        
        Args:
            video_id: ID of the video
            session: Chat session with memory
            summary: Optional stored summary information
            
        Returns:
            ConversationalRetrievalChain: Configured chain for conversation
        """
        # Initialize LLM
        logging.info(f"Initializing LLM for video {video_id} with model {config.DEFAULT_SUMMARY_MODEL} from {config.MODEL_PROVIDER}")
        llm = init_chat_model(
            model=config.DEFAULT_SUMMARY_MODEL,
            model_provider=config.MODEL_PROVIDER,
            temperature=config.TEMPERATURE
        )
        
        # Get or create retriever
        retriever = ChatChainFactory._create_retriever(video_id, summary)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CHAT_SYSTEM_TEMPLATE),
            ("human", "{question}")
        ])
        
        # Create chain
        logging.info(f"Creating ConversationalRetrievalChain for video {video_id}")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=session.memory,
            return_source_documents=True,
            verbose=config.DEBUG,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            }
        )
        
        return chain
    
    @staticmethod
    def _create_retriever(video_id: str, summary: Optional[Dict[str, Any]] = None):
        """Create a retriever with the vector store."""
        # Get vector store
        vector_store = VectorStoreManager.get_vector_store(video_id)
        
        # If no vector store and summary provided, create one
        if not vector_store and summary and summary.get("transcript_text"):
            manager = VectorStoreManager(video_id)
            vector_store = manager.get_or_create_store(summary["transcript_text"])
            
        if not vector_store:
            raise ValueError(f"No vector store available for video {video_id}")
            
        # Create base retriever
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": config.RETRIEVAL_K,
                "fetch_k": config.RETRIEVAL_FETCH_K
            }
        )
        
        # Create embeddings filter for better results
        embeddings = initalize_embedding_model()
        embeddings_filter = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=config.SIMILARITY_THRESHOLD
        )
        
        # Create contextual compression retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever
        )
        
        return retriever
    
    @staticmethod
    def create_fallback_chain(question: str, transcript_text: str):
        """Create a fallback chain using direct transcript text."""
        llm = init_chat_model(
            model=config.DEFAULT_SUMMARY_MODEL,
            model_provider=config.MODEL_PROVIDER,
            temperature=config.TEMPERATURE
        )
        
        # Create a direct prompt with transcript excerpt
        excerpt = transcript_text[:config.MAX_CONTEXT_LENGTH]
        direct_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an AI assistant that helps users understand video content.
            Here's a part of the video transcript:

            {excerpt}

            Answer the user's question based on this excerpt. If you can't answer from this excerpt,
            say that you have limited information from the transcript."""),
            ("human", "{question}")
        ])
        
        # Create simple chain
        chain = direct_prompt | llm
        
        return chain