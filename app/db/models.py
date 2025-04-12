"""
SQLAlchemy models for the YouTube Video Summarizer database.
"""

import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Boolean, JSON
from sqlalchemy.orm import relationship

from app.db.database import Base


class Video(Base):
    """Model representing a YouTube video."""
    __tablename__ = "videos"

    id = Column(String(20), primary_key=True)  # YouTube video ID
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=False)
    url = Column(String(255), nullable=False)
    audio_path = Column(String(255), nullable=True)
    video_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    transcript = relationship("Transcript", back_populates="video", uselist=False, cascade="all, delete-orphan")
    summary = relationship("Summary", back_populates="video", uselist=False, cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="video", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Video(id='{self.id}', title='{self.title}')>"


class Transcript(Base):
    """Model representing a video transcript."""
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(20), ForeignKey("videos.id", ondelete="CASCADE"), unique=True)
    text = Column(Text, nullable=False)
    file_path = Column(String(255), nullable=True)
    model = Column(String(50), nullable=True)
    segments = Column(JSON, nullable=True)  # For storing transcript segments with timestamps
    language = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="transcript")

    def __repr__(self):
        return f"<Transcript(id={self.id}, video_id='{self.video_id}')>"


class Summary(Base):
    """Model representing a video summary."""
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(20), ForeignKey("videos.id", ondelete="CASCADE"), unique=True)
    text = Column(Text, nullable=False)
    model = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="summary")

    def __repr__(self):
        return f"<Summary(id={self.id}, video_id='{self.video_id}')>"


class ChatHistory(Base):
    """Model representing chat history with a video."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(20), ForeignKey("videos.id", ondelete="CASCADE"))
    session_id = Column(String(50), nullable=False)  # To group messages by user session
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="chat_history")

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, video_id='{self.video_id}', session_id='{self.session_id}')>"