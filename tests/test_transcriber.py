"""
Tests for the audio transcriber module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from app.models.schemas import YouTubeMedia, TranscriptionConfig, TranscriptedData
from app.core.transcriber import AudioTranscriber

from dotenv import load_dotenv
load_dotenv()

@pytest.fixture
def mock_groq_client():
    """Fixture to mock the Groq client."""
    with patch('app.core.transcriber.Groq') as mock_groq:
        mock_client = mock_groq.return_value

        # Mock transcriptions API
        mock_client.audio = MagicMock()
        mock_client.audio.transcriptions = MagicMock()
        mock_client.audio.transcriptions.create = MagicMock()

        # Set up mock transcription response
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = json.dumps({
            "text": "This is a test transcript",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": "This is a test"}
            ],
            "language": "en",
            "model": "whisper-large-v3-turbo"
        })

        mock_client.audio.transcriptions.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def media():
    """Fixture to create a YouTubeMedia object."""
    return YouTubeMedia(
        video_id="test123",
        title="Test Video",
        author="Test Author",
        url="https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R",
        audio_path="/tmp/test_audio.mp3"
    )


@pytest.fixture
def transcription_config():
    """Fixture to create a TranscriptionConfig object."""
    return TranscriptionConfig(
        model="whisper-large-v3-turbo",
        language="en",
        prompt="Transcribe this YouTube video",
        response_format="verbose_json",
        temperature=0.0,
        timestamp_granularities=["segment"]
    )


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_init_transcriber():
    """Test initializing the transcriber."""
    transcriber = AudioTranscriber(TranscriptionConfig())
    assert transcriber.api_key == "test_api_key"


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
@patch('os.path.exists')
@patch('os.path.isfile')
@patch('builtins.open', new_callable=mock_open, read_data=b'test audio data')
def test_transcribe(mock_file_open, mock_isfile, mock_exists, mock_groq_client, media, transcription_config, tmp_path):
    """Test transcribing audio file."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = True

    # Create test dirs
    os.environ["TRANSCRIPTS_DIR"] = str(tmp_path)

    # Create transcriber and transcribe
    transcriber = AudioTranscriber(transcription_config)
    result = transcriber.transcribe(media)

    # Verify API was called
    mock_groq_client.audio.transcriptions.create.assert_called_once()

    # Verify result
    assert isinstance(result, YouTubeMedia)
    assert result.transcript_path is not None


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
@patch('os.path.exists')
@patch('os.path.isfile')
@patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
    "text": "This is a test transcript",
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.5, "text": "This is a test transcript"}
    ],
    "language": "en",
    "model": "whisper-large-v3-turbo"
}))
def test_get_transcripted_data_info(mock_file_open, mock_isfile, mock_exists, media):
    """Test getting transcript data info."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = True

    # Set transcript path
    media.transcript_path = "/tmp/test_transcript.json"

    # Create transcriber and get transcript data
    transcriber = AudioTranscriber(TranscriptionConfig())
    transcript_data = transcriber.get_transcripted_data_info(media)

    # Verify result
    assert isinstance(transcript_data, TranscriptedData)
    assert transcript_data.transcript_text == "This is a test transcript"
    assert len(transcript_data.segments) == 1
    assert transcript_data.language == "en"
    assert transcript_data.model == "whisper-large-v3-turbo"


@patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"})
def test_transcribe_file_not_found(media):
    """Test transcribing with non-existent audio file."""
    # Set audio path to non-existent file
    media.audio_path = "/tmp/nonexistent_file.mp3"

    # Create transcriber
    transcriber = AudioTranscriber(TranscriptionConfig())

    # Verify FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(media)