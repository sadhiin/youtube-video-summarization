"""
FastAPI application for the YouTube Video Summarizer.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from functools import lru_cache

from app.api.routes import router
from app.config import config
from app.db.database import init_db
from app.utils.caching import setup_redis_cache
from app.utils.vector_store import init_vector_store
from app.utils.logger import logging

# Create the FastAPI application
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="An API for downloading, transcribing, and summarizing YouTube videos",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    # Initialize database
    init_db()

    # Initialize vector store
    init_vector_store()
    logging.info("Vector store initialized")

    # Rebuild vector stores for videos with missing stores but available transcripts
    try:
        from app.db.database import get_db
        from app.db.crud import get_stored_summary
        from app.utils.vector_store import add_to_vector_db, get_vector_store_for_video, _VIDEO_VECTOR_STORES

        db_session = next(get_db())
        from app.db.models import Video
        videos = db_session.query(Video).all()

        logging.info(f"Found {len(videos)} videos in database, checking vector stores...")

        for video in videos:
            if not get_vector_store_for_video(video.id):
                logging.info(f"Rebuilding vector store for video {video.id}")
                # Get summary and check for transcript
                summary = get_stored_summary(db_session, video.id)
                if summary and summary.get("transcript_text"):
                    transcript_length = len(summary["transcript_text"])
                    logging.info(f"Found transcript for video {video.id} with length: {transcript_length} of transcript text")

                    vector_store = add_to_vector_db(video.id, summary["transcript_text"])
                    if vector_store:
                        logging.info(f"Successfully created vector store for video {video.id}")
                    else:
                        logging.info(f"Failed to create vector store for video {video.id}")
                else:
                    logging.info(f"No transcript text found for video {video.id}")

        logging.info(f"Vector store initialization complete. {len(_VIDEO_VECTOR_STORES)} vector stores available.")
    except Exception as e:
        logging.error(f"Error rebuilding vector stores: {e}")
        import traceback
        logging.error(traceback.format_exc())

    # Set up Redis cache if configured
    if hasattr(config, "REDIS_URL") and config.REDIS_URL:
        setup_redis_cache(config.REDIS_URL)


# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"},
    )


# Include the API router
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning basic API information."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "description": "YouTube Video Summarizer API",
    }