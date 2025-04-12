"""
Reusable UI components for the Streamlit app.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable
import datetime
import time
import re
import uuid


def header():
    """Display the application header."""
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎬 YouTube Video Summarizer")
    st.markdown("""
    Get concise AI summaries of YouTube videos and chat with their content.
    """)
    st.divider()


def sidebar():
    """Display the sidebar with app information and options."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/logo.png", width=100)
        st.title("YouTube Summarizer")

        st.markdown("## About")
        st.info("""
        This app helps you quickly understand YouTube content by:
        - Generating concise summaries
        - Allowing you to chat with the video content
        - Searching across multiple videos
        """)

        st.markdown("## Settings")
        api_url = st.text_input("API URL", value="http://localhost:8000", key="api_url")

        # Store settings in session state
        if "api_url" not in st.session_state:
            st.session_state.api_url = api_url

        st.divider()
        st.markdown("### Made By [@sadhiin](https://github.com/sadhiin) with ❤️ ")


def youtube_input():
    """
    Display a YouTube URL input field.

    Returns:
        The entered YouTube URL or None
    """
    with st.form(key="youtube_form"):
        url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.form_submit_button("Summarize")
        with col2:
            force_refresh = st.checkbox("Force refresh (ignore cache)", value=False)

    if submit and url:
        return url, force_refresh

    return None, False


def display_summary(summary: Dict[str, Any]):
    """
    Display the video summary.

    Args:
        summary: Dictionary containing summary data
    """
    st.markdown(f"## {summary['title']}")
    st.markdown(f"**Author:** {summary['author']}")

    st.markdown("### Summary")
    st.markdown(summary['summary'])

    # Store the session data
    if "current_video" not in st.session_state:
        st.session_state.current_video = {
            "id": summary["video_id"],
            "title": summary["title"],
            "author": summary["author"],
        }


def loading_spinner(message: str = "Processing..."):
    """
    Display a loading spinner with a message.

    Args:
        message: Message to display with the spinner
    """
    return st.spinner(message)


def display_error(message: str):
    """
    Display an error message.

    Args:
        message: Error message to display
    """
    st.error(message)


def display_success(message: str):
    """
    Display a success message.

    Args:
        message: Success message to display
    """
    st.success(message)


def get_random_session_id():
    """Generate a random session ID for chat."""
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    return st.session_state.chat_session_id


def chat_interface(video_id: str, chat_callback: Callable):
    """
    Display a chat interface for interacting with a video.

    Args:
        video_id: YouTube video ID
        chat_callback: Function to call when a message is sent
    """
    st.markdown("## Chat with this Video")
    st.markdown("Ask questions about the video content:")

    # Initialize or get chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the video...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                session_id = get_random_session_id()
                response = chat_callback(video_id, user_input, session_id)
                answer = response["answer"]
                st.markdown(answer)

                if "sources" in response and response["sources"]:
                    with st.expander("Sources"):
                        for i, source in enumerate(response["sources"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"```\n{source['content']}\n```")

        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


def search_interface(search_callback: Callable):
    """
    Display a search interface for finding videos by content.

    Args:
        search_callback: Function to call when a search is performed
    """
    st.markdown("## Search Videos by Content")

    with st.form(key="search_form"):
        query = st.text_input(
            "Search query",
            placeholder="Enter keywords to search across videos...",
        )
        limit = st.slider("Maximum results", min_value=1, max_value=10, value=5)
        submit = st.form_submit_button("Search")

    if submit and query:
        with st.spinner("Searching..."):
            results = search_callback(query, limit)

            if not results:
                st.info("No matching videos found.")
            else:
                st.success(f"Found {len(results)} matching videos:")

                for i, result in enumerate(results):
                    with st.expander(f"{i+1}. {result['title']} - {result['author']}"):
                        st.markdown(result['summary'])
                        if st.button("Chat with this Video", key=f"chat_btn_{i}"):
                            st.session_state.current_video = {
                                "id": result["video_id"],
                                "title": result["title"],
                                "author": result["author"],
                            }
                            st.session_state.chat_history = []
                            st.rerun()


def youtube_embed(video_id: str):
    """
    Embed a YouTube video.

    Args:
        video_id: YouTube video ID
    """
    st.markdown(f"""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}"
    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media;
    gyroscope; picture-in-picture" allowfullscreen></iframe>
    """, unsafe_allow_html=True)


def display_transcript_preview(text: str, max_length: int = 500):
    """
    Display a preview of the transcript.

    Args:
        text: Full transcript text
        max_length: Maximum length to display
    """
    if len(text) > max_length:
        preview = text[:max_length] + "..."
    else:
        preview = text

    with st.expander("Transcript Preview"):
        st.markdown(preview)
        if len(text) > max_length:
            st.markdown(f"*Transcript is {len(text)} characters long. Showing first {max_length} characters.*")
            if st.button("View Full Transcript"):
                st.markdown(text)