import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# Initialize the Groq client
client = Groq()

# Specify the path to the audio file
# filename = os.path.dirname(__file__) + "/YOUR_AUDIO.wav" # Replace with your audio file!
filename = "/teamspace/studios/this_studio/youtube-video-summarization/.downloads1744266851_Nuclear Fusion_ explained for beginners.mp3"
filename = "/teamspace/studios/this_studio/youtube-video-summarization/.downloads1744267163_Nuclear Fusion Explained.mp3"

# Open the audio file
with open(filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
      file=file, # Required audio file
      model="whisper-large-v3-turbo", # Required model to use for transcription
      prompt="Specify context or spelling",  # Optional
      response_format="verbose_json",  # Optional
      timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
      language="en",  # Optional
      temperature=0.0  # Optional
    )
    # To print only the transcription text, you'd use print(transcription.text) (here we're printing the entire transcription object to access timestamps)
    # Save the transcription to a JSON file with same base name
    current_file = os.path.basename(__file__)
    base_name = os.path.splitext(current_file)[0]+f"_{time.time()}"
    json_filename = f"{base_name}.json"
    
    with open(json_filename, 'w') as json_file:
        json.dump(transcription, json_file, indent=2, default=str)
    
    print(f"Transcription saved to {json_filename}")