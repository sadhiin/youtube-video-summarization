# YouTube Video Summarizer Tests

This directory contains test files for the YouTube Video Summarizer application.

## Setup

Install the development dependencies:

```bash
pip install -r tests/requirements_dev.txt
```

## Running Tests

To run all tests:

```bash
# From the project root
pytest
```

To run specific test files:

```bash
pytest tests/test_youtube_downloader.py
pytest tests/test_transcriber.py
pytest tests/test_summarizer.py
pytest tests/test_integration.py
```

To run tests with coverage reporting:

```bash
pytest --cov=app tests/
```

## Test Structure

The tests are organized as follows:

- `test_youtube_downloader.py`: Tests for the YouTube downloader component
- `test_transcriber.py`: Tests for the audio transcription component
- `test_summarizer.py`: Tests for the transcript summarization component
- `test_integration.py`: End-to-end integration tests

## Test Environment

The test environment is set up in `conftest.py`, which creates test directories and sets environment variables.

## Mocking

The tests use mocking to isolate components and avoid external API calls:

- YouTube API calls are mocked to avoid actual video downloads
- Groq API calls are mocked to avoid actual transcription and summarization
- File system operations use temporary paths

## Running in CI

These tests are designed to be run in CI environments. Make sure to set the GROQ_API_KEY
environment variable in your CI configuration (but use a dummy value for testing).