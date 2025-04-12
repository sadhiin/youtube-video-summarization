from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from app.config import config

def get_embedding_model():
    embedding_model = NVIDIAEmbeddings(
        model=config.VECTOR_EMBEDDING_MODEL,
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE", )
    return embedding_model