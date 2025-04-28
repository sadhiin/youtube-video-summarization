"""
Tests for the transcript summarizer module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.models.schemas import (
    YouTubeMedia,
    SummaryConfig,
    VideoSummary,
    TranscriptedData
)
from app.core.summarizer import TranscriptSummarizer
from dotenv import load_dotenv
load_dotenv()

@pytest.fixture
def mock_langchain_model():
    """Fixture to mock the langchain chat model."""
    with patch('app.core.summarizer.init_chat_model') as mock_init_model:
        # Create mock model
        mock_model = MagicMock()

        # Configure invoke to return a mock response
        mock_response = MagicMock()
        mock_response.content = "This is a summarized transcript of the video."
        mock_model.invoke.return_value = mock_response

        # Make init_chat_model return our mock model
        mock_init_model.return_value = mock_model

        yield mock_model


@pytest.fixture
def media():
    """Fixture to create a YouTubeMedia object."""
    return YouTubeMedia(
        video_id="test123",
        title="Test Video",
        author="Test Author",
        url="https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R",
        audio_path="/tmp/test_audio.mp3",
        transcript_path="/tmp/test_transcript.json"
    )


@pytest.fixture
def transcript_data():
    """Fixture to create transcript data."""
    return TranscriptedData(
        transcript_text="This is a test transcript for a video about testing. "
                     "We are testing the summarization functionality. "
                     "This text is used to test our summarizer module. "
                     "In this video, we discuss unit testing and mocking. "
                     "The main topics are pytest, unittest, and mock objects.",
        segments=[
            {"id": 0, "start": 0.0, "end": 5.0, "text": "This is a test transcript for a video about testing."},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "We are testing the summarization functionality."}
        ],
        language="en",
        model="whisper-large-v3-turbo"
    )


@pytest.fixture
def summary_config():
    """Fixture to create a SummaryConfig object."""
    return SummaryConfig(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=2048,
        num_lines=5
    )


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_init_summarizer():
    """Test initializing the summarizer."""
    summarizer = TranscriptSummarizer()
    assert summarizer.api_key == "test_api_key"


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_summarize_short_transcript(mock_langchain_model, summary_config):
    """Test summarizing a short transcript (single chunk)."""
    # Short transcript (will be processed as a single chunk)
    transcript_text = "This is a short test transcript."

    # Create summarizer and summarize
    summarizer = TranscriptSummarizer()
    summary = summarizer.summarize(transcript_text, summary_config)

    # Verify model was called
    mock_langchain_model.invoke.assert_called_once()

    # Verify result
    assert summary == "This is a summarized transcript of the video."


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
@patch('app.core.summarizer.RecursiveCharacterTextSplitter.split_documents')
def test_summarize_long_transcript(mock_split_docs, mock_langchain_model, summary_config):
    """Test summarizing a long transcript (multiple chunks)."""
    # Mock the text splitter to return multiple document chunks
    from langchain_core.documents import Document

    # Create some document chunks
    chunk1 = Document(page_content="This is the first part of a long transcript.")
    chunk2 = Document(page_content="This is the second part of a long transcript.")
    mock_split_docs.return_value = [chunk1, chunk2]

    # Long transcript (will be processed with map-reduce approach)
    long_transcript = "This is a really long transcript " * 100

    # Create summarizer and summarize
    summarizer = TranscriptSummarizer()
    summary = summarizer.summarize(long_transcript, summary_config)

    # Verify model was called multiple times (map and reduce)
    assert mock_langchain_model.invoke.call_count > 1

    # Verify result
    assert summary == "This is a summarized transcript of the video."


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_create_summary(mock_langchain_model, media, transcript_data, summary_config):
    """Test creating a complete video summary."""
    # Create summarizer and generate summary
    summarizer = TranscriptSummarizer()
    video_summary = summarizer.create_summary(media, transcript_data, summary_config)

    # Verify result
    assert isinstance(video_summary, VideoSummary)
    assert video_summary.media_info == media
    assert video_summary.summary == "This is a summarized transcript of the video."
    assert video_summary.transcript_text == transcript_data.transcript_text
    assert video_summary.transcript_segments == transcript_data.segments
    assert video_summary.language == transcript_data.language
    assert video_summary.model == transcript_data.model


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_summarize_with_keywords(mock_langchain_model, summary_config):
    """Test summarizing with selective keywords."""
    # Set selective keywords
    summary_config.selective_keywords = "testing, pytest, mock"

    # Create summarizer and summarize
    summarizer = TranscriptSummarizer()
    transcript_text = "This is a test transcript about pytest and mocking."
    summarizer.summarize(transcript_text, summary_config)

    # Verify the prompt includes keywords (can't easily check the exact prompt,
    # but we can verify the model was called)
    mock_langchain_model.invoke.assert_called_once()