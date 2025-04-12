"""
Main entry point for the YouTube Video Summarizer application.
"""

import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

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


def init_directories():
    """Create necessary directories for storing data."""
    # Use absolute paths from the project root
    os.makedirs(os.path.join(config.BASE_DIR, "data", "downloads"), exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR, "data", "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR, "data", "summaries"), exist_ok=True)


def save_summary(summary: VideoSummary, output_file: str = None):
    """Save the summary to a JSON file."""
    if output_file is None:
        output_dir = Path(os.path.join(config.BASE_DIR, "data", "summaries"))
        output_dir.mkdir(parents=True, exist_ok=True)
        video_id = summary.media_info.video_id or "unknown"
        output_file = output_dir / f"{video_id}_summary.json"
    else:
        output_file = Path(output_file)

    # Convert to dict for serialization
    summary_dict = summary.model_dump()

    # Serialize nested models
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)

    print(f"Summary saved to: {output_file}")
    return output_file


def summarize_youtube_video(
    url: str,
    groq_model: str = "deepseek-r1-distill-qwen-32b",
    output_file: str = None
) -> VideoSummary:
    """
    Process a YouTube video: download audio, transcribe, and summarize.

    Args:
        url: YouTube video URL
        groq_model: Groq language model to use for summarization
        output_file: Optional file path to save the summary

    Returns:
        VideoSummary object
    """
    # Get the project root directory path from config
    project_root = config.BASE_DIR

    # 1. Download YouTube audio
    download_config = YouTubeDownloadConfig(
        url=url,
        media_type=MediaType.AUDIO,
        output_directory=str(Path(project_root) / "data" / "downloads")
    )


    downloader = YouTubeDownloader(download_config)
    print(f"Downloading audio from: {url}")
    media = downloader.download()
    print("Download complete.")
    media_info = downloader.get_media_info()
    # 2. Transcribe audio
    transcription_config = TranscriptionConfig()

    transcriber = AudioTranscriber()
    print("Transcribing audio...")
    media = transcriber.transcribe(media, transcription_config)

    # 3. Extract transcript text
    transcript_text = transcriber.get_transcript_text(media)

    # 4. Summarize transcript
    summary_config = SummaryConfig(
        model=groq_model,
        temperature=0.0,
        max_tokens=1024,
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
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-32b",
                        help="Groq language model for summarization")
    parser.add_argument("--output", help="Output file path for the summary")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize directories
    init_directories()

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