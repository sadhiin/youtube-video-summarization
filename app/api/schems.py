from pydantic import BaseModel
from typing import Optional, List, Dict, Any
class VideoRequest(BaseModel):
    """Model for requesting video summarization."""
    url: str
    model: Optional[str] = "llama-3.3-70b-versatile"
    force_refresh: Optional[bool] = False
    num_lines: Optional[int] = 5
    selective_keywords: Optional[str] = None


class ChatRequest(BaseModel):
    """Model for chat requests."""
    video_id: str
    message: str


class ChatResponse(BaseModel):
    """Model for chat responses."""
    answer: str
    sources: List[Dict[str, Any]] = []


class SummaryResponse(BaseModel):
    """Model for summary responses."""
    video_id: str
    title: str
    author: str
    summary: str
    audio_available: bool
    transcript_available: bool
    cached: bool = False


class SearchRequest(BaseModel):
    """Model for similar video search requests."""
    query: str
    limit: Optional[int] = 5