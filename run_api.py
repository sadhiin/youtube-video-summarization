"""
FastAPI server entry point for the YouTube Video Summarizer.
"""

import os
import argparse
import uvicorn
from dotenv import load_dotenv

from app.config import config


def main():
    """Run the FastAPI server."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YouTube Video Summarizer API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # Initialize directories
    config.initialize()

    # Print startup info
    print(f"Starting {config.APP_NAME} API server v{config.APP_VERSION}")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Binding to: {args.host}:{args.port}")

    # Run the server
    uvicorn.run(
        "app.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=config.LOG_LEVEL.lower() if hasattr(config, "LOG_LEVEL") else "info"
    )


if __name__ == "__main__":
    main()