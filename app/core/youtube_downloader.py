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
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use pathlib for reliable path joining
        return os.path.join(output_dir, f"{base_filename}.{extension}")

    def download_video(self) -> str:
        try:
            video_stream = self.yt.streams.get_highest_resolution()
            full_output_path = self._get_filename("mp4")
            
            # Split path into directory and filename components
            output_dir = os.path.dirname(full_output_path)
            filename = os.path.basename(full_output_path)
            
            logging.info(f"Downloading video: {self.yt.title}")
            if self.config.save_file:
                video_stream.download(output_path=output_dir, filename=filename)
                
                # Verify file exists
                if not os.path.exists(full_output_path):
                    logging.error(f"File not found at expected path: {full_output_path}")
                else:
                    logging.info(f"Video saved to: {full_output_path}")
                
                return full_output_path
            else:
                logging.info("Video processed in memory (not saved)")
                return ""

        except Exception as e:
            logging.error(f"Error downloading video: {str(e)}")
            raise


    def download_audio(self) -> str:
        try:
            audio_stream = self.yt.streams.filter(only_audio=True).order_by('abr').last()
            full_output_path = self._get_filename("mp3")
            
            # Split path into directory and filename components
            output_dir = os.path.dirname(full_output_path)
            filename = os.path.basename(full_output_path)
            
            logging.debug(f"Output directory: {output_dir}")
            logging.debug(f"Filename: {filename}")
            logging.info(f"Downloading audio: {self.yt.title}")
            
            if self.config.save_file:
                # Correctly specify output_path and filename separately
                audio_stream.download(output_path=output_dir, filename=filename)
                
                # Verify file exists after download
                if not os.path.exists(full_output_path):
                    logging.error(f"File not found at expected path: {full_output_path}")
                else:
                    logging.info(f"Audio saved to: {full_output_path}")
                
                return full_output_path
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
        print("**********Meida info**********")
        print(media_info)
        logging.info(f"Media info: {media_info}")
        return media_info