"""
Main Streamlit application for YouTube Video Summarizer.
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import os
from app.frontend.api_client import ApiClient
from app.frontend.components import (
    header, sidebar, youtube_input, display_summary,
    loading_spinner, display_error, display_success,
    chat_interface, search_interface, youtube_embed,
    display_transcript_preview
)


load_dotenv()

def init_session_state():
    """Initialize session state variables."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = ApiClient(st.session_state.get("api_url", "http://localhost:8000"))

    if "current_view" not in st.session_state:
        st.session_state.current_view = "home"

    if "current_video" not in st.session_state:
        st.session_state.current_video = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def process_youtube_url(url: str, force_refresh: bool = False, num_lines: int = 5, selective_keywords: Optional[str] = None):
    """
    Process a YouTube URL to get a summary.

    Args:
        url: YouTube URL
        force_refresh: Whether to force regeneration of the summary
        num_lines: Desired number of lines for the summary
        selective_keywords: Optional comma-separated keywords to focus on

    Returns:
        Video summary or error message
    """
    client = st.session_state.api_client

    try:
        # Extract video ID
        video_id = client.extract_video_id(url)
        if not video_id:
            return {"error": "Invalid YouTube URL format"}

        # Request summary
        with loading_spinner("Requesting video summary..."):
            result = client.summarize_video(url, force_refresh, num_lines, selective_keywords)

        # If processing in background, wait for completion
        if "summary" in result and result["summary"] == "Processing in background. Please check back shortly.":
            with loading_spinner("Processing video. This may take a minute..."):
                result = client.wait_for_summary(result["video_id"])

        return result

    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}


def handle_chat(video_id: str, message: str, session_id: Optional[str] = None):
    """
    Handle chat interaction with a video.

    Args:
        video_id: YouTube video ID
        message: User message
        session_id: Session ID for continuing a conversation

    Returns:
        Chat response
    """
    client = st.session_state.api_client

    try:
        response = client.chat_with_video(video_id, message, session_id)
        return response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {"answer": f"Sorry, an error occurred: {str(e)}", "sources": []}


def handle_search(query: str, limit: int = 5):
    """
    Handle search for videos by content.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        List of matching videos
    """
    client = st.session_state.api_client

    try:
        results = client.search_videos(query, limit)
        return results
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []


def home_view():
    """Display the home view with YouTube URL input."""
    st.markdown("## Get Started")
    st.markdown("""
    Enter a YouTube URL to generate a summary. You can then chat with the video content or search across multiple videos.
    """)

    url, force_refresh, num_lines, selective_keywords = youtube_input()

    if url:
        # Process the URL
        result = process_youtube_url(url, force_refresh, num_lines, selective_keywords)

        if "error" in result:
            display_error(result["error"])
        else:
            display_success("Video processed successfully!")

            # Switch to video view
            st.session_state.current_video = {
                "id": result["video_id"],
                "title": result["title"],
                "author": result["author"],
            }
            st.session_state.current_view = "video"

            # Clear chat history for new video
            st.session_state.chat_history = []

            st.rerun()

    # Sample videos section
    st.markdown("## Sample Videos")
    st.markdown("Try these example videos:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Nuclear Fusion Explained")
        st.markdown("A short educational video about nuclear fusion")
        if st.button("Try this video", key="sample1"):
            sample_url = "https://youtu.be/Hy8fB32GZoc"
            st.session_state.sample_url = sample_url
            st.rerun()

    with col2:
        st.markdown("### Metaverse Explained")
        st.markdown("An explanation of the metaverse concept")
        if st.button("Try this video", key="sample2"):
            sample_url = "https://youtu.be/4S4C11V2Lvc"
            st.session_state.sample_url = sample_url
            st.rerun()

    # Process sample URL if selected
    if "sample_url" in st.session_state:
        url = st.session_state.sample_url
        del st.session_state.sample_url
        # Use default values for samples
        result = process_youtube_url(url, False, 5, None)

        if "error" in result:
            display_error(result["error"])
        else:
            display_success("Video processed successfully!")

            # Switch to video view
            st.session_state.current_video = {
                "id": result["video_id"],
                "title": result["title"],
                "author": result["author"],
            }
            st.session_state.current_view = "video"

            # Clear chat history for new video
            st.session_state.chat_history = []

            st.rerun()

    # Search section
    st.markdown("## Search Existing Videos")
    st.markdown("Search for videos by content:")

    search_interface(handle_search)


def video_view():
    """Display the video view with summary and chat interface."""
    if not st.session_state.current_video:
        st.session_state.current_view = "home"
        st.rerun()
        return

    video = st.session_state.current_video

    # Add navigation
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_view = "home"
            st.rerun()

    with col3:
        if st.button("Search"):
            st.session_state.current_view = "search"
            st.rerun()

    # Get full summary
    client = st.session_state.api_client
    try:
        summary = client.get_summary(video["id"])

        if "error" in summary:
            display_error(summary["error"])
            st.session_state.current_view = "home"
            st.rerun()
            return

        # Display video embed
        youtube_embed(video["id"])

        # Display summary
        display_summary(summary)

        # Display transcript preview if available
        if "transcript_text" in summary and summary["transcript_text"]:
            display_transcript_preview(summary["transcript_text"])

        # Display chat interface
        chat_interface(video["id"], handle_chat)

    except Exception as e:
        display_error(f"Error loading video: {str(e)}")
        st.session_state.current_view = "home"
        st.rerun()


def search_view():
    """Display the search view."""
    # Add navigation
    if st.button("← Back"):
        if st.session_state.current_video:
            st.session_state.current_view = "video"
        else:
            st.session_state.current_view = "home"
        st.rerun()

    st.markdown("## Search Videos by Content")
    st.markdown("""
    Search for videos across the database by their content.
    The system will find videos with transcripts most relevant to your query.
    """)

    search_interface(handle_search)


def main():
    """Main application entry point."""
    # Set up the UI
    header()
    sidebar()
    with st.sidebar:
        debug_mode = st.checkbox("Debug Mode", value=False)
    # Initialize session state
    init_session_state()

    # Display the appropriate view
    if st.session_state.current_view == "home":
        home_view()
    elif st.session_state.current_view == "video":
        video_view()
    elif st.session_state.current_view == "search":
        search_view()
    else:
        # Default to home view
        st.session_state.current_view = "home"
        home_view()

    def chat_callback(video_id, message, session_id):
        response = api.chat_with_video(video_id, message, session_id)

        if debug_mode:
            st.write("Debug Information:")
            st.write(f"- Video ID: {video_id}")
            st.write(f"- Session ID: {session_id}")
            st.write(f"- Sources retrieved: {len(response.get('sources', []))}")
            if response.get('sources'):
                st.write("First source sample:")
                st.code(response['sources'][0]['content'][:200])

        return response

if __name__ == "__main__":
    main()