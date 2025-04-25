"""
API routes for the YouTube Video Summarizer application.
"""

import traceback
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from pydantic import BaseModel

from app.models.schemas import (
    YouTubeDownloadConfig,
    MediaType,
)
from app.api.schems import (
    VideoRequest,
    SummaryResponse,
    ChatRequest,
    ChatResponse,
    SearchRequest,
)
from app.core.youtube_downloader import YouTubeDownloader
from app.core.transcriber import AudioTranscriber
from app.core.summarizer import TranscriptSummarizer
from app.db.crud import get_stored_summary, store_summary, video_exists
from app.db.database import get_db, DBSession
from app.utils.logger import logging

# Import new modules
from app.core.vectorstore.manager import VectorStoreManager
from app.core.chat.handler import ChatHandler

router = APIRouter(prefix="/api/v1", tags=["youtube"])


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
):
    """
    Summarize a YouTube video by URL.

    - If the video has been processed before, returns cached result
    - If force_refresh is True, reprocesses the video
    - Runs processing in background for new videos
    """
    # Extract video ID from URL
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Check if video already exists and cached result can be used
    if not request.force_refresh and video_exists(db, video_id):
        stored_summary = get_stored_summary(db, video_id)
        logging.debug(f"Stored summary: {stored_summary}")
        if stored_summary:
            return SummaryResponse(
                video_id=video_id,
                title=stored_summary["title"],
                author=stored_summary["author"],
                summary=stored_summary["summary"],
                audio_available=bool(stored_summary["audio_path"]),
                transcript_available=bool(stored_summary["transcript_path"]),
                cached=True,
            )

    try:
        config = YouTubeDownloadConfig(url=request.url, media_type=MediaType.AUDIO)
        downloader = YouTubeDownloader(config)
        media_info = downloader.get_media_info()

        # background processing
        background_tasks.add_task(
            process_video_in_background,
            url=request.url,
            model=request.model,
            num_lines=request.num_lines,
            selective_keywords=request.selective_keywords,
            db=db,
        )

        return SummaryResponse(
            video_id=video_id,
            title=media_info.title,
            author=media_info.author,
            summary="Processing in background. Please check back shortly.",
            audio_available=False,
            transcript_available=False,
        )

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@router.get("/summary/{video_id}", response_model=SummaryResponse)
async def get_summary(
    video_id: str = Path(..., description="YouTube video ID"),
    db: DBSession = Depends(get_db),
):
    """Get the summary for a processed video by ID."""
    stored_summary = get_stored_summary(db, video_id)
    logging.debug(f"Stored summary: {stored_summary}")

    if not stored_summary:
        raise HTTPException(
            status_code=404, detail="Video not found or not yet processed"
        )

    return SummaryResponse(
        video_id=video_id,
        title=stored_summary["title"],
        author=stored_summary["author"],
        summary=(
            stored_summary["summary"]
            if "summary" in stored_summary
            else "Processing in background. Please check back shortly."
        ),
        audio_available=bool(stored_summary["audio_path"]),
        transcript_available=bool(stored_summary["transcript_path"]),
        cached=True,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_video(chat_request: ChatRequest, db: DBSession = Depends(get_db)):
    """Chat with a processed video using the transcript as context."""
    # Validate video and transcript availability
    stored_summary = get_stored_summary(db, chat_request.video_id)
    if not stored_summary or not stored_summary.get("transcript_text"):
        raise HTTPException(
            status_code=404, detail="Video not found or transcript not available"
        )

    try:
        # Use the new ChatHandler
        response = ChatHandler.get_chat_response(
            video_id=chat_request.video_id,
            message=chat_request.message,
            db=db,
            session_id=chat_request.session_id,
        )

        return response
    except ValueError as e:
        logging.error(f"Value error in chat: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


@router.post("/search", response_model=List[SummaryResponse])
async def search_videos(search_request: SearchRequest, db: DBSession = Depends(get_db)):
    """Search for similar videos based on transcript content."""
    try:
        vector_store_manager = (
            VectorStoreManager.init_vector_stores()
        )  # this will load the vector store for all videos in the database

        similar_videos = VectorStoreManager.search_similar_videos(
            query=search_request.query, limit=search_request.limit
        )

        # Get summaries for each video
        results = []
        for video_id in similar_videos:
            stored_summary = get_stored_summary(db, video_id)
            if stored_summary:
                results.append(
                    SummaryResponse(
                        video_id=video_id,
                        title=stored_summary["title"],
                        author=stored_summary["author"],
                        summary=(
                            stored_summary["summary"]
                            if "summary" in stored_summary
                            else "Summary not available"
                        ),
                        audio_available=bool(stored_summary["audio_path"]),
                        transcript_available=bool(stored_summary["transcript_path"]),
                        cached=True,
                    )
                )

        return results
    except Exception as e:
        logging.error(f"Error searching videos: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error searching videos: {str(e)}")


# Helper functions
def extract_video_id(url: str) -> Optional[str]:
    """Extract the video ID from a YouTube URL."""
    import re

    # YouTube URL patterns
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:watch\?v=)([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


async def process_video_in_background(
    url: str,
    model: str,
    num_lines: int = 5,
    selective_keywords: Optional[str] = None,
    db: DBSession = None,
):
    """Process a video in the background and store results in DB."""
    from app.main import summarize_youtube_video

    try:
        # Process the video
        summary = summarize_youtube_video(
            url=url,
            groq_model=model,
            num_lines=num_lines,
            selective_keywords=selective_keywords,
        )

        if not summary.transcript_text:
            logging.warning(
                f"No transcript text generated for video {summary.media_info.video_id}"
            )
        else:
            logging.info(
                f"Generated transcript of length {len(summary.transcript_text)} for video {summary.media_info.video_id}"
            )

        store_summary(db, summary)
        logging.info(
            f"Stored summary in database for video {summary.media_info.video_id}"
        )

        if summary.transcript_text:
            # Use the new VectorStoreManager
            video_id = summary.media_info.video_id
            manager = VectorStoreManager(video_id)

            # Get or create vector store with retry logic built in
            vector_store = manager.get_or_create_store(summary.transcript_text)

            if vector_store:
                logging.info(
                    f"Successfully added transcript to vector store for video {video_id}"
                )
            else:
                logging.error(
                    f"Failed to add transcript to vector store for video {video_id}"
                )
                logging.info(f"Creating emergency vector store for video {video_id}")

                # Try emergency store as last resort
                vector_store = manager.create_emergency_store(summary.transcript_text)

                if vector_store:
                    logging.info(
                        f"Successfully created emergency vector store for video {video_id}"
                    )
                else:
                    logging.error(
                        f"Could not create any vector store for video {video_id}"
                    )
        else:
            logging.warning(
                f"No transcript text available to add to vector store for video {summary.media_info.video_id}"
            )
    except Exception as e:
        logging.error(f"Background processing error: {str(e)}")
        logging.error(traceback.format_exc())
