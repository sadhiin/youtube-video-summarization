from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import ChatConfig
from app.utils.logger import logging
from retry import retry

class VectorStoreManager:
    """Manages creation and retrieval of vector stores with retry logic."""
    
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.config = ChatConfig()

    @retry(tries=ChatConfig.VECTOR_STORE_RETRIES, 
           delay=ChatConfig.VECTOR_STORE_RETRY_DELAY,
           backoff=ChatConfig.VECTOR_STORE_BACKOFF,
           logger=logging)
    def get_or_create_store(self, transcript: str) -> FAISS:
        """Get or create vector store with fallback mechanisms."""
        from app.utils.vector_store import get_vector_store_for_video
        
        # Try existing store first
        if store := get_vector_store_for_video(self.video_id):
            return store
            
        # Create new store with proper chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        
        try:
            docs = self._create_documents(transcript, splitter)
            return FAISS.from_documents(docs, self._get_embeddings())
        except Exception as e:
            logging.error(f"Vector store creation failed: {str(e)}")
            return self._create_emergency_store(transcript)

    def _create_documents(self, transcript: str, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
        """Create documents with proper metadata."""
        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": "transcript",
                    "chunk_id": i,
                    "video_id": self.video_id
                }
            )
            for i, chunk in enumerate(splitter.split_text(transcript))
        ]

    def _create_emergency_store(self, transcript: str) -> FAISS:
        """Fallback store creation with simplified chunks."""
        emergency_text = transcript[:self.config.EMERGENCY_CHUNK_LIMIT]
        return FAISS.from_documents(
            [Document(
                page_content=emergency_text,
                metadata={"source": "emergency_transcript"}
            )],
            self._get_embeddings()
        )

    def _get_embeddings(self):
        """Initialize embeddings model."""
        from app.embeddings.get_embedding_model import initalize_embedding_model
        return initalize_embedding_model()