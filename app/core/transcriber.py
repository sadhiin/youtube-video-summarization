"""
Module for transcribing audio files using Groq's API.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from groq import Groq

from app.models.schemas import YouTubeMedia, TranscriptionConfig, TranscriptedData
from app.utils.logger import logging
from app.config import config


class AudioTranscriber:
    """Class to handle audio transcription operations."""

    def __init__(
        self, transcribe_config: TranscriptionConfig, api_key: Optional[str] = None
    ):
        """
        Initialize the transcriber with API key.

        Args:
            api_key: Groq API key (if None, will try to get from environment)
        """
        self.transcribe_config = transcribe_config
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set it in .env file or pass directly."
            )

        self.client = Groq(api_key=self.api_key)

    def transcribe(self, media: YouTubeMedia) -> YouTubeMedia:
        """
        Transcribe the audio from a YouTubeMedia object.

        Args:
            media: YouTubeMedia object with audio_path set
            config: Configuration for transcription

        Returns:
            Updated YouTubeMedia with transcript_path set
        """
        if not media.audio_path:
            raise ValueError("Audio path not found in media object")

        if not os.path.exists(media.audio_path) or not os.path.isfile(media.audio_path):
            raise FileNotFoundError(f"Audio file not found at {media.audio_path}")

        # Define transcript path
        audio_file_path = Path(media.audio_path)
        transcript_dir = Path(config.TRANSCRIPTS_DIR)
        transcript_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = transcript_dir / f"{audio_file_path.stem}.json"

        # Perform transcription
        logging.info(f"Transcribing audio file: {media.audio_path}")

        with open(media.audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=(audio_file_path.name, audio_file.read()),
                model=self.transcribe_config.model,
                prompt=self.transcribe_config.prompt,
                response_format=self.transcribe_config.response_format,
                timestamp_granularities=self.transcribe_config.timestamp_granularities,
                temperature=self.transcribe_config.temperature,
            )

        # Save the transcription to a JSON file
        logging.info(f"Saving transcription to: {transcript_path}")

        with open(transcript_path, "w", encoding="utf-8") as f:
            if hasattr(transcription, "model_dump_json"):
                f.write(transcription.model_dump_json(indent=4))
            else:
                json.dump(transcription, f, indent=2, default=str)

        # Update the media object
        media.transcript_path = str(transcript_path)
        logging.info("Transcription complete.")

        return media

    def get_transcripted_data_info(self, media: YouTubeMedia) -> TranscriptedData:
        """
        Get the full transcript data with segments.

        Args:
            media: YouTubeMedia object with transcript_path set

        Returns:
            Full transcript data with segments
        """
        transcript_data = TranscriptedData(
            transcript_text="none",
            segments=[],
            file_path=None,
            model=None,
            language=None,
        )

        if not media.transcript_path:
            raise ValueError("Transcript path not found in media object")
        logging.debug(f"Loading transcript from: {media.transcript_path}")
        if not os.path.exists(media.transcript_path) or not os.path.isfile(
            media.transcript_path
        ):
            raise FileNotFoundError(
                f"Transcript file not found at {media.transcript_path}"
            )
        with open(media.transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract segments based on response format
        if isinstance(data, dict):
            # For verbose_json format
            if "text" in transcript_data:
                transcript_data.transcript_text = data["text"]
            if "segments" in data:
                transcript_data.segments = data["segments"]
            if "language" in data:
                transcript_data.language = data["language"]
            if "model" in data:
                transcript_data.model = data["model"]
        return transcript_data
