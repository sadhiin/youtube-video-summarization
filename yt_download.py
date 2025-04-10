
# loved song: https://youtu.be/jqOuWRtgsXU?si=5ZLLfNZMwyZUZY2o
# Install with: pip install pytubefix
# from pytubefix import YouTube

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

from pytubefix import YouTube
from pytubefix.cli import on_progress

def download_audio(url):
    yt = YouTube(url, on_progress_callback=on_progress)
    
    # Get the highest quality audio stream
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
    
    # Download the audio
    print(f"Downloading: {yt.title}")
    audio_stream.download()
    print("Download complete!")

video_url = input("Enter YouTube URL: ")
download_audio(video_url)
