"""
Launcher script for the YouTube Video Summarizer Streamlit app.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Launch the Streamlit app with command line options."""
    parser = argparse.ArgumentParser(description="YouTube Video Summarizer Streamlit App")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    parser.add_argument("--server-port", type=int, default=8000, help="Port where the FastAPI server is running")
    parser.add_argument("--api-url", help="URL of the API server (default: http://localhost:8000)")
    args = parser.parse_args()

    # Get the absolute path of the app directory
    app_dir = Path(__file__).parent.absolute()
    app_path = app_dir / "app" / "frontend" / "streamlit_app.py"

    # Set up environment variables
    env = os.environ.copy()

    # Set API URL if provided
    if args.api_url:
        api_url = args.api_url
    else:
        api_url = f"http://localhost:{args.server_port}"

    env["API_URL"] = api_url

    # Add the project root to PYTHONPATH to fix import issues
    env["PYTHONPATH"] = str(app_dir) + os.pathsep + env.get("PYTHONPATH", "")

    # Print startup info
    print(f"Starting YouTube Video Summarizer Streamlit app on port {args.port}")
    print(f"API server is expected to be running at: {api_url}")
    print(f"Added {app_dir} to PYTHONPATH for imports")

    # Construct the command to run Streamlit
    cmd = [
        "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false",
    ]

    # Run Streamlit app
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("Streamlit app stopped")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()