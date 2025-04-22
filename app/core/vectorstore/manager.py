import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache


from retry import retry
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.embeddings.get_embedding_model import initalize_embedding_model

from app.config import config
from app.utils.logger import logging
from app.utils.error_handling import handle_vector_store_error

_VIDEO_VECTOR_STORES = {}

class VectorStoreManager:
    """Manages creation, retrieval, and operation of vector stores.
    """

    def __init__(self, video_id:str):
        """Initialize the vector store manager for a specific video.
        Args:
            video_id (str): The ID of the video for which to manage the vector store.
        """
        self.video_id = video_id
        self.vector_dir = config.DATA_DIR / "vector_indices"
        self.video_dir = self.vector_dir / video_id
    
    @retry(tries=config.VECTOR_STORE_RETRIES, 
           delay=config.VECTOR_STORE_RETRY_DELAY,
           backoff=config.VECTOR_STORE_BACKOFF,
           logger=logging)
    def get_or_create_store(self, transcript: str) -> Optional[FAISS]:
        """Get existing or create new vector store with comprehensive error handling."""
        # Check if already loaded in memory
        if store := _VIDEO_VECTOR_STORES.get(self.video_id):
            return store
            
        # Check if exists on disk
        if self._exists_on_disk():
            return self._load_from_disk()
            
        return self._create_new_store(transcript)
    
    def _exists_on_disk(self) -> bool:
        """Check if vector store exists on disk."""
        index_path = self.video_dir / "index"
        return index_path.exists()
    
    def _load_from_disk(self) -> Optional[FAISS]:
        """Load vector store from disk."""
        try:
            embeddings = initalize_embedding_model()
            index_path = self.video_dir / "index"
            vector_store = FAISS.load_local(
                str(index_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Cache in memory
            _VIDEO_VECTOR_STORES[self.video_id] = vector_store
            logging.info(f"Loaded vector index for video {self.video_id}")
            
            return vector_store
        except Exception as e:
            logging.error(f"Error loading vector index for video {self.video_id}: {str(e)}")
            return None
    
    def _create_new_store(self, transcript: str) -> Optional[FAISS]:
        """Create new vector store from transcript."""
        if not transcript or len(transcript.strip()) < 10:
            logging.error(f"Error: Transcript for video {self.video_id} is too short or empty")
            return None
            
        try:
            # Initialize directories
            os.makedirs(self.video_dir, exist_ok=True)
            
            # Create chunked documents
            documents = self._create_documents(transcript)
            if not documents:
                raise ValueError("No chunks were created from transcript")
                
            # Create vector store
            embeddings = initalize_embedding_model()
            vector_store = FAISS.from_documents(documents, embeddings)
            
            # Test the store
            self._test_vector_store(vector_store)
            
            # Save to disk
            index_path = self.video_dir / "index"
            vector_store.save_local(str(index_path))
            
            # Cache in memory
            _VIDEO_VECTOR_STORES[self.video_id] = vector_store
            
            logging.info(f"Successfully created vector store with {len(documents)} chunks for video {self.video_id}")
            return vector_store
        except Exception as e:
            return handle_vector_store_error(e, self.video_id, transcript)

        
    def _create_documents(self, transcript: str) -> List[Document]:
        """Create documents with metadata from transcript."""
        # Strip and clean the text
        cleaned_text = transcript.strip()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = text_splitter.split_text(cleaned_text)
        logging.info(f"Split transcript into {len(chunks)} chunks for vector storage")
        
        # Create documents with metadata
        return [
            Document(
                page_content=chunk,
                metadata={
                    "video_id": self.video_id,
                    "chunk_id": i,
                    "source": "transcript",
                    "chunk_length": len(chunk)
                }
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _test_vector_store(self, vector_store: FAISS) -> None:
        """Test vector store with simple query."""
        test_results = vector_store.similarity_search("test query", k=1)
        if not test_results:
            logging.warning(f"Vector store created for {self.video_id} but test query returned no results")
    
    def create_emergency_store(self, transcript: str) -> Optional[FAISS]:
        """Create emergency vector store with simplified approach."""
        try:
            logging.info(f"Creating emergency vector store for video {self.video_id}")
            
            # Use a small piece of the transcript for emergency
            emergency_text = transcript[:config.EMERGENCY_CHUNK_LIMIT]
            
            embeddings = initalize_embedding_model()
            document = Document(
                page_content=emergency_text,
                metadata={
                    "source": "emergency_transcript", 
                    "video_id": self.video_id
                }
            )
            
            vector_store = FAISS.from_documents([document], embeddings)
            
            # Cache in memory
            _VIDEO_VECTOR_STORES[self.video_id] = vector_store
            
            return vector_store
        except Exception as e:
            logging.error(f"Emergency vector store creation failed: {str(e)}")
            return None
    
    @staticmethod
    def remove_store(video_id: str) -> bool:
        """Remove a vector store for a video."""
        try:
            # Remove from memory
            if video_id in _VIDEO_VECTOR_STORES:
                del _VIDEO_VECTOR_STORES[video_id]
            
            # Remove from disk
            vector_dir = config.DATA_DIR / "vector_indices" / video_id
            if vector_dir.exists():
                import shutil
                shutil.rmtree(vector_dir)
                
            return True
        except Exception as e:
            logging.error(f"Error removing vector store for video {video_id}: {str(e)}")
            return False

    @staticmethod
    @lru_cache(maxsize=128)
    def get_vector_store(video_id: str) -> Optional[FAISS]:
        """Get vector store for a video with caching."""
        return _VIDEO_VECTOR_STORES.get(video_id)

    @staticmethod
    def init_vector_stores():
        """Initialize all vector stores from disk."""
        vector_dir = config.DATA_DIR / "vector_indices"
        os.makedirs(vector_dir, exist_ok=True)
        
        # Skip if not a directory or doesn't exist
        if not vector_dir.exists() or not vector_dir.is_dir():
            return
            
        embeddings = initalize_embedding_model()
        
        # Load each index
        for video_dir in vector_dir.iterdir():
            if video_dir.is_dir():
                video_id = video_dir.name
                index_path = video_dir / "index"
                
                if index_path.exists():
                    try:
                        vector_store = FAISS.load_local(
                            str(index_path), 
                            embeddings, 
                            allow_dangerous_deserialization=True
                        )
                        _VIDEO_VECTOR_STORES[video_id] = vector_store
                        logging.info(f"Loaded vector index for video {video_id}")
                    except Exception as e:
                        logging.error(f"Error loading vector index for video {video_id}: {str(e)}")
