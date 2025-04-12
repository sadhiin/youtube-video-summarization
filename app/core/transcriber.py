"""
Module for transcribing audio files using Groq's API.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from groq import Groq

from app.models.schemas import YouTubeMedia, TranscriptionConfig
from app.utils.logger import logging
from app.config import config

class AudioTranscriber:
    """Class to handle audio transcription operations."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the transcriber with API key.

        Args:
            api_key: Groq API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set it in .env file or pass directly.")

        self.client = Groq(api_key=self.api_key)

    def transcribe(self, media: YouTubeMedia, config: TranscriptionConfig) -> YouTubeMedia:
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

        if not os.path.exists(media.audio_path):
            raise FileNotFoundError(f"Audio file not found at {media.audio_path}")

        # Define transcript path
        audio_file_path = Path(media.audio_path)
        transcript_dir = Path(os.path.join(config.BASE_DIR, "data", "transcripts"))
        transcript_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = transcript_dir / f"{audio_file_path.stem}.json"

        # Perform transcription
        logging.info(f"Transcribing audio file: {media.audio_path}")

        with open(media.audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=(audio_file_path.name, audio_file.read()),
                model=config.model,
                prompt=config.prompt,
                response_format=config.response_format,
                # timestamp_granularities=config.timestamp_granularities,
                # language=config.language,
                temperature=config.temperature
            )

        # Save the transcription to a JSON file
        logging.info(f"Saving transcription to: {transcript_path}")

        with open(transcript_path, "w", encoding="utf-8") as f:
            if hasattr(transcription, "model_dump_json"):
                # For Pydantic models
                f.write(transcription.model_dump_json(indent=2))
            else:
                # For normal dictionaries or objects
                json.dump(transcription, f, indent=2, default=str)

        # Update the media object
        media.transcript_path = str(transcript_path)
        logging.info("Transcription complete.")

        return media

    def get_transcript_text(self, media: YouTubeMedia) -> str:
        """
        Get the text-only transcript from a transcribed media.

        Args:
            media: YouTubeMedia object with transcript_path set

        Returns:
            Plain text of the transcript
        """
        if not media.transcript_path:
            raise ValueError("Transcript path not found in media object")

        with open(media.transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Extract text based on response format
        if isinstance(transcript_data, dict):
            # For verbose_json format
            if "text" in transcript_data:
                return transcript_data["text"]

        # Fallback for different formats
        return str(transcript_data)

    def get_transcript_segments(self, media: YouTubeMedia) -> Dict[str, Any]:
        """
        Get the full transcript data with segments.

        Args:
            media: YouTubeMedia object with transcript_path set

        Returns:
            Full transcript data with segments
        """
        if not media.transcript_path:
            raise ValueError("Transcript path not found in media object")

        with open(media.transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)