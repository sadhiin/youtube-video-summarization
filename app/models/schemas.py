"""
Data models for the YouTube summarizer application.
"""
import time
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from app.config import config
import os


class MediaType(str, Enum):
    """Types of media that can be downloaded."""
    VIDEO = "video"
    AUDIO = "audio"
    BOTH = "both"


class YouTubeDownloadConfig(BaseModel):
    """Configuration for YouTube download operations."""
    url: str
    media_type: MediaType = MediaType.AUDIO
    output_filename: Optional[str] = None
    output_directory: str = os.path.join(config.BASE_DIR, "data/downloads")
    save_file: bool = True

    @field_validator('url')
    def validate_youtube_url(cls, v):
        if 'youtube.com' not in v and 'youtu.be' not in v:
            raise ValueError('URL must be a valid YouTube URL')
        return v


class YouTubeMedia(BaseModel):
    """Model to store YouTube media metadata and file paths."""
    video_id: str = ""
    title: str
    author: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None

    model_config = {"from_attributes": True}


class TranscriptionConfig(BaseModel):
    """Configuration for transcription operations."""
    model: str = "whisper-large-v3-turbo"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: str = "verbose_json"
    temperature: float = 0.0
    # timestamp_granularities: List[str] = ["segment"]


class SummaryConfig(BaseModel):
    """Configuration for summarization operations."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    chunk_size: int = 4000
    chunk_overlap: int = 400


class VideoSummary(BaseModel):
    """Model for storing video summary information."""
    media_info: YouTubeMedia
    summary: str
    transcript_text: Optional[str] = None
    transcript_segments: Optional[List[Dict[str, Any]]] = None
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))