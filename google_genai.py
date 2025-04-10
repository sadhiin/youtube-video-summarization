import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_video(video_url: str) -> str:
    
    response = client.models.generate_content(
        model='models/gemini-2.0-flash',
        contents=types.Content(
            parts=[
                types.Part(text='Can you summarize this video?'),
                types.Part(
                    file_data=types.FileData(file_uri=video_url)
                )
            ]
        )
    )
    return response

if __name__=="__main__":
    video_url = "https://youtu.be/piPbnKdve9M?si=2XgMT3oWjqNYQ_jJ"
    summary = summarize_video(video_url)
    print("Summary:", summary)