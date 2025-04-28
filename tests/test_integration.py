"""
Integration tests for the YouTube Video Summarizer.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from app.main import summarize_youtube_video
from app.models.schemas import MediaType, VideoSummary


@pytest.fixture
def mock_youtube_downloader():
    """Fixture to mock the YouTubeDownloader class."""
    with patch('app.main.YouTubeDownloader') as mock_downloader_class:
        # Configure mock instance and download method
        mock_instance = mock_downloader_class.return_value

        # Create a media object for the mock to return
        from app.models.schemas import YouTubeMedia
        media = YouTubeMedia(
            video_id="test123",
            title="Test Integration Video",
            author="Test Author",
            url="https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R",
            audio_path="/tmp/test_audio.mp3"
        )

        mock_instance.download.return_value = media
        mock_instance.get_media_info.return_value = media

        yield mock_instance


@pytest.fixture
def mock_audio_transcriber():
    """Fixture to mock the AudioTranscriber class."""
    with patch('app.main.AudioTranscriber') as mock_transcriber_class:
        # Configure mock instance and transcribe method
        mock_instance = mock_transcriber_class.return_value

        # Configure mock to update media with transcript path
        def mock_transcribe(media):
            media.transcript_path = "/tmp/test_transcript.json"
            return media

        mock_instance.transcribe.side_effect = mock_transcribe

        # Configure get_transcripted_data_info method
        from app.models.schemas import TranscriptedData
        transcript_data = TranscriptedData(
            transcript_text="This is a mock transcript for integration testing.",
            segments=[{"id": 0, "start": 0.0, "end": 5.0, "text": "This is a mock transcript."}],
            language="en",
            model="whisper-large-v3-turbo"
        )

        mock_instance.get_transcripted_data_info.return_value = transcript_data

        yield mock_instance


@pytest.fixture
def mock_transcript_summarizer():
    """Fixture to mock the TranscriptSummarizer class."""
    with patch('app.main.TranscriptSummarizer') as mock_summarizer_class:
        # Configure mock instance and create_summary method
        mock_instance = mock_summarizer_class.return_value

        # Configure create_summary method
        def mock_create_summary(media, transcript_data, config):
            from app.models.schemas import VideoSummary
            return VideoSummary(
                media_info=media,
                summary="This is an integration test summary of the video.",
                transcript_text=transcript_data.transcript_text,
                transcript_segments=transcript_data.segments,
                language=transcript_data.language,
                model=transcript_data.model
            )

        mock_instance.create_summary.side_effect = mock_create_summary

        yield mock_instance


@pytest.fixture
def mock_save_summary():
    """Fixture to mock the save_summary function."""
    with patch('app.main.save_summary') as mock_save:
        mock_save.return_value = "/tmp/test123_summary.json"
        yield mock_save


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_summarize_youtube_video(
    mock_youtube_downloader,
    mock_audio_transcriber,
    mock_transcript_summarizer,
    mock_save_summary
):
    """Test the end-to-end YouTube video summarization process."""
    # Call the main function with a test URL
    summary = summarize_youtube_video(
        url="https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R",
        groq_model="llama-3.3-70b-versatile",
        num_lines=5
    )

    # Check that all mock components were called
    mock_youtube_downloader.download.assert_called_once()
    mock_audio_transcriber.transcribe.assert_called_once()
    mock_audio_transcriber.get_transcripted_data_info.assert_called_once()
    mock_transcript_summarizer.create_summary.assert_called_once()
    mock_save_summary.assert_called_once()

    # Verify the final result
    assert isinstance(summary, VideoSummary)
    assert summary.summary == "This is an integration test summary of the video."
    assert summary.media_info.video_id == "test123"


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_summarize_youtube_video_with_keywords(
    mock_youtube_downloader,
    mock_audio_transcriber,
    mock_transcript_summarizer,
    mock_save_summary
):
    """Test summarization with selective keywords."""
    # Call the main function with keywords
    summarize_youtube_video(
        url="https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R",
        groq_model="llama-3.3-70b-versatile",
        num_lines=5,
        selective_keywords="testing, integration"
    )

    # Get the summary config that was passed to create_summary
    call_args = mock_transcript_summarizer.create_summary.call_args
    summary_config = call_args[0][2]  # args[2] should be the summary_config

    # Verify selective_keywords were passed correctly
    assert summary_config.selective_keywords == "testing, integration"


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_summarize_youtube_video_with_custom_output(
    mock_youtube_downloader,
    mock_audio_transcriber,
    mock_transcript_summarizer,
    mock_save_summary
):
    """Test summarization with custom output file."""
    # Call the main function with custom output file
    custom_output = "/tmp/custom_summary.json"
    summarize_youtube_video(
        url="https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R",
        groq_model="llama-3.3-70b-versatile",
        output_file=custom_output
    )

    # Verify save_summary was called with the custom output file
    mock_save_summary.assert_called_once_with(mock_save_summary.call_args[0][0], custom_output)