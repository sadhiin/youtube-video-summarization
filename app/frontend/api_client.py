"""
API client for communicating with the YouTube Video Summarizer backend.
"""

import json
import requests
import time
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin
from app.config import config

class ApiClient:
    """Client for interacting with the YouTube Video Summarizer API."""

    def __init__(self, base_url: str = config.PUBLIC_URL):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.api_base = urljoin(base_url, "/api/v1/")

    def _url(self, endpoint: str) -> str:
        """Get the full URL for an endpoint."""
        return urljoin(self.api_base, endpoint)

    def summarize_video(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Request a video summary.

        Args:
            url: YouTube video URL
            force_refresh: Whether to force regeneration of the summary

        Returns:
            Dictionary with video ID, title, and summary status
        """
        response = requests.post(
            self._url("summarize"),
            json={
                "url": url,
                "force_refresh": force_refresh
            }
        )

        response.raise_for_status()
        return response.json()

    def get_summary(self, video_id: str) -> Dict[str, Any]:
        """
        Get the summary for a processed video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with video details and summary
        """
        response = requests.get(self._url(f"summary/{video_id}"))

        if response.status_code == 404:
            return {"error": "Video not found or not yet processed"}

        response.raise_for_status()
        return response.json()

    def wait_for_summary(self, video_id: str, timeout: int = 300, interval: int = 5) -> Dict[str, Any]:
        """
        Wait for a video summary to complete processing.

        Args:
            video_id: YouTube video ID
            timeout: Maximum seconds to wait
            interval: Seconds between checks

        Returns:
            Dictionary with video details and summary
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            summary = self.get_summary(video_id)

            # Check if still processing
            if "error" in summary or summary.get("summary") == "Processing in background. Please check back shortly.":
                time.sleep(interval)
                continue

            return summary

        # Timeout occurred
        return {"error": "Timeout waiting for summary to complete"}

    def chat_with_video(self, video_id: str, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Chat with a video based on its transcript.

        Args:
            video_id: YouTube video ID
            message: User message/question
            session_id: Session ID for continuing a conversation

        Returns:
            Dictionary with answer and sources
        """
        payload = {
            "video_id": video_id,
            "message": message
        }

        if session_id:
            payload["session_id"] = session_id

        response = requests.post(
            self._url("chat"),
            json=payload
        )

        response.raise_for_status()
        return response.json()

    def search_videos(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for videos by content.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of video summaries matching the query
        """
        response = requests.post(
            self._url("search"),
            json={
                "query": query,
                "limit": limit
            }
        )

        response.raise_for_status()
        return response.json()

    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from a URL.

        Args:
            url: YouTube URL

        Returns:
            Video ID or None if extraction fails
        """
        import re
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened
            r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embedded videos
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'  # Standard watch URL
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None