"""
CRUD operations for YouTube Video Summarizer database.
"""

from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
import datetime
import json

from app.db.models import Video, Transcript, Summary, ChatHistory
from app.models.schemas import YouTubeMedia, VideoSummary
from app.utils.logger import logging

def video_exists(db: Session, video_id: str) -> bool:
    """Check if a video exists in the database."""
    return db.query(Video).filter(Video.id == video_id).first() is not None


def get_video(db: Session, video_id: str) -> Optional[Video]:
    """Get a video by ID."""
    return db.query(Video).filter(Video.id == video_id).first()


def create_video(db: Session, media: YouTubeMedia) -> Video:
    """Create a new video entry."""
    video = Video(
        id=media.video_id,
        title=media.title,
        author=media.author,
        url="",  # This should be properly set
        audio_path=media.audio_path,
        video_path=media.video_path
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def create_transcript(db: Session, video_id: str, transcript_text: str, file_path: Optional[str] = None,
                    model: Optional[str] = None, segments: Optional[Dict[str, Any]] = None,
                    language: Optional[str] = None) -> Transcript:
    """Create a transcript for a video."""
    transcript = Transcript(
        video_id=video_id,
        text=transcript_text,
        file_path=file_path,
        model=model,
        segments=segments,
        language=language
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return transcript


def create_summary(db: Session, video_id: str, summary_text: str, model: Optional[str] = None) -> Summary:
    """Create a summary for a video."""
    summary = Summary(
        video_id=video_id,
        text=summary_text,
        model=model
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary


def add_chat_message(db: Session, video_id: str, session_id: str, message: str,
                    response: str) -> ChatHistory:
    """Add a chat message to the history."""
    chat_entry = ChatHistory(
        video_id=video_id,
        session_id=session_id,
        message=message,
        response=response
    )
    db.add(chat_entry)
    db.commit()
    db.refresh(chat_entry)
    return chat_entry


def get_chat_history(db: Session, video_id: str, session_id: str, limit: int = 10) -> List[ChatHistory]:
    """Get chat history for a video and session."""
    return db.query(ChatHistory).filter(
        ChatHistory.video_id == video_id,
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.created_at.desc()).limit(limit).all()


def get_stored_summary(db: Session, video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a stored summary for a video, combining data from multiple tables.

    Returns a dictionary with all summary information.
    """
    video = get_video(db, video_id)
    if not video:
        logging.error(f"Video with ID {video_id} not found in the database.")
        return None

    result = {
        "video_id": video.id,
        "title": video.title,
        "author": video.author,
        "audio_path": video.audio_path,
        "video_path": video.video_path,
        "transcript_path": None,
        "summary": None,
        "transcript_text": None
    }

    # Get transcript if available
    if video.transcript:
        result["transcript_path"] = video.transcript.file_path
        result["transcript_text"] = video.transcript.text

    # Get summary if available
    if video.summary:
        result["summary"] = video.summary.text

    return result


def store_summary(db: Session, summary: VideoSummary) -> None:
    """
    Store a complete video summary in the database.

    This creates or updates entries in videos, transcripts, and summaries tables.
    """
    # First, check if video exists
    video = get_video(db, summary.media_info.video_id)

    if not video:
        # Create video entry
        video = create_video(db, summary.media_info)

    # Store transcript if available
    if summary.transcript_text:
        # Check if transcript exists
        if not video.transcript:
            # Parse segments if available
            segments = None
            if summary.transcript_segments:
                segments = summary.transcript_segments

            # Create transcript
            create_transcript(
                db=db,
                video_id=video.id,
                transcript_text=summary.transcript_text,
                file_path=summary.media_info.transcript_path,
                segments=segments
            )

    # Store summary if available
    if summary.summary and not video.summary:
        create_summary(
            db=db,
            video_id=video.id,
            summary_text=summary.summary
        )

    # Commit any pending changes
    db.commit()