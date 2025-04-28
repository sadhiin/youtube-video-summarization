"""
Configuration for pytest tests.
"""

import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables and directories."""
    # Create test data directories
    test_data_dir = Path("test_data")
    test_downloads_dir = test_data_dir / "downloads"
    test_transcripts_dir = test_data_dir / "transcripts"
    test_summaries_dir = test_data_dir / "summaries"

    # Create directories if they don't exist
    test_data_dir.mkdir(exist_ok=True)
    test_downloads_dir.mkdir(exist_ok=True)
    test_transcripts_dir.mkdir(exist_ok=True)
    test_summaries_dir.mkdir(exist_ok=True)

    # Set environment variables for testing
    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "test_api_key")
    os.environ["DATA_DIR"] = str(test_data_dir)
    os.environ["DOWNLOADS_DIR"] = str(test_downloads_dir)
    os.environ["TRANSCRIPTS_DIR"] = str(test_transcripts_dir)
    os.environ["SUMMARIES_DIR"] = str(test_summaries_dir)
    os.environ["ENVIRONMENT"] = "development"

    yield

    import shutil
    shutil.rmtree(test_data_dir)


@pytest.fixture(scope="session")
def test_video_url():
    """Return a test YouTube video URL."""
    return "https://youtu.be/V3TUEeB0kW0?si=-InVol0JhtWji-6R"


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return Path(os.environ.get("DATA_DIR", "test_data"))


@pytest.fixture(scope="session")
def test_downloads_dir():
    """Return the test downloads directory."""
    return Path(os.environ.get("DOWNLOADS_DIR", "test_data/downloads"))


@pytest.fixture(scope="session")
def test_transcripts_dir():
    """Return the test transcripts directory."""
    return Path(os.environ.get("TRANSCRIPTS_DIR", "test_data/transcripts"))


@pytest.fixture(scope="session")
def test_summaries_dir():
    """Return the test summaries directory."""
    return Path(os.environ.get("SUMMARIES_DIR", "test_data/summaries"))