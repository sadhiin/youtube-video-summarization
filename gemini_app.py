# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    st.error("Failed to initialize Gemini client. Please check your API key.")
    st.stop()

def summarize_video(video_url: str, num_lines: int, selective_words: str = "") -> str:
    try:
        prompt = f"Please summarize this video in approximately {num_lines} lines."
        if selective_words.strip():
            prompt += f" Focus on content related to the following words: {selective_words}."
        
        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=types.Content(
                parts=[
                    types.Part(text=prompt),
                    types.Part(
                        file_data=types.FileData(file_uri=video_url)
                    )
                ]
            )
        )
        
        summary = response.candidates[0].content.parts[0].text
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🎥")
    st.title("YouTube Video Summarizer with Gemini 2.0")
    st.markdown("Paste a YouTube video link and customize your summary using the Gemini 2.0 Flash model.")

    with st.form(key="video_summary_form"):
        video_url = st.text_input("YouTube Video URL", placeholder="https://youtu.be/...")
        num_lines = st.number_input("Desired number of lines for summary", min_value=1, max_value=20, value=5)
        selective_words = st.text_input("Selective words (optional, comma-separated)", placeholder="e.g., technology, AI, innovation")
        submit_button = st.form_submit_button(label="Generate Summary")

    if submit_button:
        if not video_url:
            st.error("Please provide a YouTube video URL.")
        elif not video_url.startswith(("https://youtu.be/", "https://www.youtube.com/")):
            st.error("Please provide a valid YouTube URL.")
        else:
            with st.spinner("Generating summary..."):
                summary = summarize_video(video_url, int(num_lines), selective_words)
                if summary.startswith("Error"):
                    st.error(summary)
                else:
                    st.subheader("Video Summary")
                    st.write(summary)

if __name__ == "__main__":
    main()