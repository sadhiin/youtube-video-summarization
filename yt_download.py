
# # loved song: https://youtu.be/jqOuWRtgsXU?si=5ZLLfNZMwyZUZY2o
# # Install with: pip install pytubefix

# from pytubefix import YouTube
# from pytubefix.cli import on_progress

# def download_video(url):
#     try:
#         yt = YouTube(url)
#         video = yt.streams.get_highest_resolution()
#         video.download()
#         print(f"Downloaded: {yt.title}")
#     except Exception as e:
#         print(f"Error: {str(e)}")

# video_url = input("Enter YouTube URL: ")
# download_video(video_url)


# def download_audio(url):
#     yt = YouTube(url, on_progress_callback=on_progress)
    
#     # Get the highest quality audio stream
#     audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
    
#     # Download the audio
#     print(f"Downloading: {yt.title}")
#     audio_stream.download()
#     print("Download complete!")

# video_url = input("Enter YouTube URL: ")
# download_audio(video_url)

"""
YouTube Video and Audio Downloader
Required packages: pytubefix, pydantic
"""

import os
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, validator
from pytubefix import YouTube
from pytubefix.cli import on_progress

class MediaType(str, Enum):
    """Types of media that can be downloaded."""
    VIDEO = "video"
    AUDIO = "audio"
    BOTH = "both"

class YouTubeDownloadConfig(BaseModel):
    """Configuration for YouTube download operations."""
    url: str
    media_type: MediaType = MediaType.VIDEO
    output_filename: Optional[str] = None
    output_directory: str = "./downloads"
    save_file: bool = True
    
    @validator('url')
    def validate_youtube_url(cls, v):
        if 'youtube.com' not in v and 'youtu.be' not in v:
            raise ValueError('URL must be a valid YouTube URL')
        return v

class YouTubeMedia(BaseModel):
    """Model to store YouTube media metadata and file paths."""
    video_id: str
    title: str
    author: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    
    class Config:
        orm_mode = True  # For future ORM integration

class YouTubeDownloader:
    """Class to handle downloading YouTube videos and audio."""
    
    def __init__(self, config: YouTubeDownloadConfig):
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
        """Generate filename based on config or timestamp."""
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
        """Download video and return the file path."""
        try:
            video_stream = self.yt.streams.get_highest_resolution()
            output_path = self._get_filename("mp4")
            
            print(f"Downloading video: {self.yt.title}")
            if self.config.save_file:
                video_stream.download(filename=output_path)
                print(f"Video saved to: {output_path}")
                return output_path
            else:
                print("Video processed in memory (not saved)")
                return ""
                
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            raise
    
    def download_audio(self) -> str:
        """Download audio and return the file path."""
        try:
            audio_stream = self.yt.streams.filter(only_audio=True).order_by('abr').last()
            output_path = self._get_filename("mp3")
            
            print(f"Downloading audio: {self.yt.title}")
            if self.config.save_file:
                audio_stream.download(filename=output_path)
                print(f"Audio saved to: {output_path}")
                return output_path
            else:
                print("Audio processed in memory (not saved)")
                return ""
                
        except Exception as e:
            print(f"Error downloading audio: {str(e)}")
            raise
            
    def download(self) -> YouTubeMedia:
        """Download media based on config settings."""
        media_info = self.get_media_info()
        
        if self.config.media_type in [MediaType.VIDEO, MediaType.BOTH]:
            media_info.video_path = self.download_video()
            
        if self.config.media_type in [MediaType.AUDIO, MediaType.BOTH]:
            media_info.audio_path = self.download_audio()
            
        return media_info

def main():
    """Simple command-line interface for YouTube downloader."""
    print("YouTube Downloader")
    
    url = input("Enter YouTube URL: ")
    media_type = input("Download type (video, audio, both) [default: video]: ").lower() or "video"
    
    if media_type not in ["video", "audio", "both"]:
        print("Invalid type. Using 'video' as default.")
        media_type = "video"
        
    save_file = input("Save file? (y/n) [default: y]: ").lower() != 'n'
    
    if save_file:
        output_filename = input("Output filename (leave empty for auto-generated): ").strip() or None
    else:
        output_filename = None
        
    config = YouTubeDownloadConfig(
        url=url,
        media_type=media_type,
        output_filename=output_filename,
        save_file=save_file
    )
    
    try:
        downloader = YouTubeDownloader(config)
        media_info = downloader.download()
        
        print(f"\nDownload Summary:")
        print(f"Title: {media_info.title}")
        print(f"Author: {media_info.author}")
        
        if media_info.video_path:
            print(f"Video saved to: {media_info.video_path}")
        if media_info.audio_path:
            print(f"Audio saved to: {media_info.audio_path}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()