
# loved song: https://youtu.be/jqOuWRtgsXU?si=5ZLLfNZMwyZUZY2o
# Install with: pip install pytubefix
from pytubefix import YouTube

def download_video(url):
    try:
        yt = YouTube(url)
        video = yt.streams.get_highest_resolution()
        video.download()
        print(f"Downloaded: {yt.title}")
    except Exception as e:
        print(f"Error: {str(e)}")

video_url = input("Enter YouTube URL: ")
download_video(video_url)
