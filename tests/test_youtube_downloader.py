"""
Tests for the YouTube downloader module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.models.schemas import MediaType, YouTubeDownloadConfig, YouTubeMedia
from app.core.youtube_downloader import YouTubeDownloader


@pytest.fixture
def mock_youtube():
    """Fixture to mock the YouTube class."""
    with patch('app.core.youtube_downloader.YouTube') as mock_yt:
        # Configure the mock YouTube instance
        mock_yt_instance = mock_yt.return_value
        mock_yt_instance.title = "Test Video"
        mock_yt_instance.author = "Test Author"
        mock_yt_instance.video_id = "test123"
        mock_yt_instance.watch_url = "https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R"

        # Mock streams
        mock_yt_instance.streams = MagicMock()
        mock_video_stream = MagicMock()
        mock_audio_stream = MagicMock()

        mock_yt_instance.streams.get_highest_resolution.return_value = mock_video_stream
        mock_yt_instance.streams.filter.return_value.order_by.return_value.last.return_value = mock_audio_stream

        yield mock_yt


@pytest.fixture
def download_config():
    """Fixture to create a download configuration."""
    return YouTubeDownloadConfig(
        url="https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R",
        media_type=MediaType.AUDIO,
        output_directory="test_data/downloads"
    )


def test_get_media_info(mock_youtube, download_config):
    """Test extracting media info from YouTube video."""
    downloader = YouTubeDownloader(download_config)
    media_info = downloader.get_media_info()

    assert isinstance(media_info, YouTubeMedia)
    assert media_info.video_id == "test123"
    assert media_info.title == "Test Video"
    assert media_info.author == "Test Author"
    assert media_info.url == "https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R"


def test_download_audio(mock_youtube, download_config, tmp_path):
    """Test downloading audio file."""
    # Set up temporary directory for output
    test_output_dir = os.path.join(tmp_path, "downloads")
    download_config.output_directory = test_output_dir

    # Configure the mock audio stream
    mock_yt = mock_youtube.return_value
    mock_audio_stream = mock_yt.streams.filter.return_value.order_by.return_value.last.return_value

    # Make download "create" a file
    def mock_download(**kwargs):
        os.makedirs(kwargs['output_path'], exist_ok=True)
        output_file = os.path.join(kwargs['output_path'], kwargs['filename'])
        with open(output_file, 'w') as f:
            f.write("Mock audio content")
        return output_file

    mock_audio_stream.download.side_effect = mock_download

    # Create the downloader and download
    downloader = YouTubeDownloader(download_config)
    audio_path = downloader.download_audio()

    # Verify download was called
    mock_audio_stream.download.assert_called_once()

    # Verify file was created
    assert os.path.exists(audio_path)

    # Clean up temp file if needed
    if os.path.exists(audio_path):
        os.remove(audio_path)


def test_download_video(mock_youtube, download_config, tmp_path):
    """Test downloading video file."""
    # Set up temporary directory for output and switch to video type
    test_output_dir = os.path.join(tmp_path, "downloads")
    download_config.output_directory = test_output_dir
    download_config.media_type = MediaType.VIDEO

    # Configure the mock video stream
    mock_yt = mock_youtube.return_value
    mock_video_stream = mock_yt.streams.get_highest_resolution.return_value

    # Make download "create" a file
    def mock_download(**kwargs):
        os.makedirs(kwargs['output_path'], exist_ok=True)
        output_file = os.path.join(kwargs['output_path'], kwargs['filename'])
        with open(output_file, 'w') as f:
            f.write("Mock video content")
        return output_file

    mock_video_stream.download.side_effect = mock_download

    # Create the downloader and download
    downloader = YouTubeDownloader(download_config)
    video_path = downloader.download_video()

    # Verify download was called
    mock_video_stream.download.assert_called_once()

    # Verify file was created
    assert os.path.exists(video_path)

    # Clean up temp file if needed
    if os.path.exists(video_path):
        os.remove(video_path)


def test_download_both(mock_youtube, download_config, tmp_path):
    """Test downloading both audio and video."""
    # Set up temporary directory for output
    test_output_dir = os.path.join(tmp_path, "downloads")
    download_config.output_directory = test_output_dir
    download_config.media_type = MediaType.BOTH

    # Configure the mock streams
    mock_yt = mock_youtube.return_value
    mock_audio_stream = mock_yt.streams.filter.return_value.order_by.return_value.last.return_value
    mock_video_stream = mock_yt.streams.get_highest_resolution.return_value

    # Make download "create" files
    def mock_download(**kwargs):
        os.makedirs(kwargs['output_path'], exist_ok=True)
        output_file = os.path.join(kwargs['output_path'], kwargs['filename'])
        with open(output_file, 'w') as f:
            f.write("Mock content")
        return output_file

    mock_audio_stream.download.side_effect = mock_download
    mock_video_stream.download.side_effect = mock_download

    # Create the downloader and download
    downloader = YouTubeDownloader(download_config)
    media = downloader.download()

    # Verify downloads were called
    mock_audio_stream.download.assert_called_once()
    mock_video_stream.download.assert_called_once()

    # Verify we got back a YouTubeMedia object with both paths
    assert isinstance(media, YouTubeMedia)
    assert os.path.exists(media.audio_path)
    assert os.path.exists(media.video_path)