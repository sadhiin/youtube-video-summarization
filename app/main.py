"""
Main entry point for the YouTube Video Summarizer application.
"""

import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from app.models.schemas import (
    MediaType,
    YouTubeDownloadConfig,
    TranscriptionConfig,
    SummaryConfig,
    VideoSummary
)
from app.core.youtube_downloader import YouTubeDownloader
from app.core.transcriber import AudioTranscriber
from app.core.summarizer import TranscriptSummarizer
from app.config import config
from app.utils.logger import logging

def save_summary(summary: VideoSummary, output_file: str = None):
    """Save the summary to a JSON file."""
    if output_file is None:
        output_dir = Path(config.SUMMARIES_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_id = summary.media_info.video_id or "unknown"
        output_file = output_dir / f"{video_id}_summary.json"
    else:
        output_file = Path(output_file)

    summary_dict = summary.model_dump()

    # Serialize nested models
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)

    logging.info(f"Summary saved to: {output_file}")
    return output_file


def summarize_youtube_video(
    url: str,
    groq_model: str = config.DEFAULT_SUMMARY_MODEL,
    num_lines: int = 5,
    selective_keywords: Optional[str] = None,
    output_file: str = None
) -> VideoSummary:
    """
    Process a YouTube video: download audio, transcribe, and summarize.

    Args:
        url: YouTube video URL
        groq_model: Groq language model to use for summarization
        num_lines: The desired number of lines for the summary
        selective_keywords: Optional comma-separated keywords to focus on in the summary
        output_file: Optional file path to save the summary

    Returns:
        VideoSummary object
    """

    # 1. Download YouTube audio
    download_config = YouTubeDownloadConfig(
        url=url,
        media_type=MediaType.AUDIO,
        output_directory=str(config.DOWNLOADS_DIR)
    )


    downloader = YouTubeDownloader(download_config)
    logging.info(f"Downloading audio from: {url}")

    media = downloader.download()
    logging.info("Download complete.")

    # media_info = downloader.get_media_info()
        
    transcription_config = TranscriptionConfig()

    transcriber = AudioTranscriber()

    logging.info("Transcribing audio...")

    media = transcriber.transcribe(media, transcription_config)

    # 3. Extract transcript text
    transcript_text = transcriber.get_transcript_text(media)

    # 4. Summarize transcript
    summary_config = SummaryConfig(
        model=groq_model,
        temperature=0.0,
        max_tokens=2048,
        num_lines=num_lines,
        selective_keywords=selective_keywords
    )

    summarizer = TranscriptSummarizer()
    print("Generating summary...")
    summary = summarizer.create_summary(media, transcript_text, summary_config)

    # 5. Save summary
    if output_file:
        save_summary(summary, output_file)
    else:
        save_summary(summary)

    return summary


def main():
    """Main function to run the application from command line."""
    parser = argparse.ArgumentParser(description="YouTube Video Summarizer")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq language model for summarization")
    parser.add_argument("--output", help="Output file path for the summary")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Process the video
    summary = summarize_youtube_video(args.url, args.model, args.output)

    # Print the summary
    print("\n" + "=" * 80)
    print(f"Summary of '{summary.media_info.title}' by {summary.media_info.author}")
    print("=" * 80)
    print(summary.summary)
    print("=" * 80)


if __name__ == "__main__":
    main()