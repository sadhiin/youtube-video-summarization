"""
YouTube video and audio downloader module.
"""

import os
import time
from enum import Enum
from typing import Optional
from pathlib import Path

from pytubefix import YouTube
from pytubefix.cli import on_progress

from app.models.schemas import MediaType, YouTubeDownloadConfig, YouTubeMedia
from app.utils.logger import logging

class YouTubeDownloader:
    """Class to handle downloading YouTube videos and audio."""

    def __init__(self, config: YouTubeDownloadConfig):
        """
        Initialize the YouTube downloader with configuration.

        Args:
            config: Configuration for download operations
        """
        self.config = config
        self.yt = YouTube(config.url, on_progress_callback=on_progress)

    def get_media_info(self) -> YouTubeMedia:
        """Extract metadata from YouTube video."""
        return YouTubeMedia(
            video_id=self.yt.video_id,
            title=self.yt.title,
            author=self.yt.author
        )

    def _get_filename(self, extension: str) -> str:
        """
        Generate filename based on config or timestamp.

        Args:
            extension: File extension (e.g., 'mp3', 'mp4')

        Returns:
            Full file path
        """
        if self.config.output_filename:
            base_filename = self.config.output_filename
        else:
            # Use timestamp and video title for filename
            timestamp = int(time.time())
            # Clean up title to make it filesystem safe
            safe_title = "".join([c if c.isalnum() or c in " -_." else "_" for c in self.yt.title])
            base_filename = f"{timestamp}_{safe_title[:50]}"

        # Ensure output directory exists
        os.makedirs(self.config.output_directory, exist_ok=True)

        return os.path.join(self.config.output_directory, f"{base_filename}.{extension}")

    def download_video(self) -> str:
        """
        Download video and return the file path.

        Returns:
            Path to downloaded video file or empty string if not saved
        """
        try:
            video_stream = self.yt.streams.get_highest_resolution()
            output_path = self._get_filename("mp4")

            logging.info(f"Downloading video: {self.yt.title}")
            if self.config.save_file:
                video_stream.download(filename=output_path)
                logging.info(f"Video saved to: {output_path}")
                return output_path
            else:
                logging.info("Video processed in memory (not saved)")
                return ""

        except Exception as e:
            logging.error(f"Error downloading video: {str(e)}")
            raise

    def download_audio(self) -> str:
        """
        Download audio and return the file path.

        Returns:
            Path to downloaded audio file or empty string if not saved
        """
        try:
            audio_stream = self.yt.streams.filter(only_audio=True).order_by('abr').last()
            output_path = self._get_filename("mp3")
            logging.debug(f"Output path:{output_path}")
            logging.info(f"Downloading audio: {self.yt.title}")
            if self.config.save_file:
                audio_stream.download(filename=output_path)
                logging.info(f"Audio saved to: {output_path}")
                return output_path
            else:
                logging.info("Audio processed in memory (not saved)")
                return ""

        except Exception as e:
            logging.error(f"Error downloading audio: {str(e)}")
            raise

    def download(self) -> YouTubeMedia:
        """
        Download media based on config settings.

        Returns:
            YouTubeMedia object with metadata and file paths
        """
        media_info = self.get_media_info()

        if self.config.media_type in [MediaType.VIDEO, MediaType.BOTH]:
            media_info.video_path = self.download_video()

        if self.config.media_type in [MediaType.AUDIO, MediaType.BOTH]:
            media_info.audio_path = self.download_audio()

        return media_info