# YouTube Video Summarizer

A Python application that downloads YouTube videos, transcribes them using Groq's Whisper model, and generates concise summaries using LLM models.

## Features

- Download YouTube videos (audio only or video)
- Transcribe audio with Groq's Whisper API
- Generate summaries using Groq's LLM models
- Save and retrieve transcripts and summaries
- **FastAPI-based REST API** for web integration
- **Streamlit user interface** for easy interaction
- **Database integration** with SQLAlchemy (SQLite by default, PostgreSQL optional)
- **Chat with video content** using Langchain memory
- **FAISS vector database** for semantic search
- **Caching** (Redis or in-memory) for performance optimization
- Command-line interface for easy use

## Installation

### Prerequisites

- Python 3.11+
- Groq API key
- Redis (optional, for caching)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sadhiin/youtube-video-summarization
   cd youtube-video-summarizer
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Streamlit Web Interface

The easiest way to use the application is through the Streamlit web interface:

1. First, start the API server:
   ```bash
   python run_api.py
   ```

2. Then, in a separate terminal, start the Streamlit app:
   ```bash
   python run_streamlit.py
   ```

3. Open your browser and navigate to http://localhost:8501

The Streamlit interface provides an intuitive way to:
- Enter YouTube URLs for summarization
- View video summaries and transcripts
- Chat with video content using natural language
- Search across multiple videos by content

### Command Line

To summarize a YouTube video via command line:

```bash
python -m app.main "https://www.youtube.com/watch?v=VIDEO_ID"
```

Additional options:
```bash
python -m app.main "https://www.youtube.com/watch?v=VIDEO_ID" --model "llama3-70b-8192" --output "path/to/output.json"
```

### REST API

To start the API server:

```bash
python run_api.py
```

or with custom settings:

```bash
python run_api.py --host 127.0.0.1 --port 8080 --reload
```

The API will be available at `http://localhost:8000/` with these endpoints:

- `POST /api/v1/summarize`: Submit a YouTube URL for processing
- `GET /api/v1/summary/{video_id}`: Get summary for a processed video
- `POST /api/v1/chat`: Ask questions about a video's content
- `POST /api/v1/search`: Search for videos by content

API documentation available at `http://localhost:8000/docs`

### As a Python Package

You can also use the summarizer programmatically:

```python
from app.main import summarize_youtube_video

# Basic usage
summary = summarize_youtube_video("https://www.youtube.com/watch?v=VIDEO_ID")

# Custom options
summary = summarize_youtube_video(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    groq_model="llama-3.3-70b-versatile",
    output_file="path/to/output.json"
)

# Print the summary
print(summary.summary)
```

## Project Structure

```
youtube-summarizer/
│
├── app/                           # Main application code
│   ├── __init__.py                # Package initialization
│   ├── main.py                    # CLI application entry point
│   ├── config.py                  # Configuration management
│   │
│   ├── api/                       # API functionality
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   └── routes.py              # API endpoints
│   │
│   ├── frontend/                  # Streamlit frontend
│   │   ├── __init__.py
│   │   ├── streamlit_app.py       # Main Streamlit application
│   │   ├── api_client.py          # Client for API communication
│   │   └── components.py          # Reusable UI components
│   │
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── youtube_downloader.py  # YouTube download functionality
│   │   ├── transcriber.py         # Speech-to-text functionality
│   │   ├── summarizer.py          # LLM summarization
│   │   └── chat_handler.py        # Chat functionality with LLM memory
│   │
│   ├── db/                        # Database functionality
│   │   ├── __init__.py
│   │   ├── database.py            # Database connection setup
│   │   ├── models.py              # SQLAlchemy models
│   │   └── crud.py                # CRUD operations
│   │
│   ├── models/                    # Data models
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic models
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── helpers.py             # Helper functions
│       ├── vector_store.py        # FAISS vector store implementation
│       └── caching.py             # Redis/in-memory caching utilities
│
├── data/                          # Data storage
│   ├── downloads/                 # Downloaded audio files
│   ├── transcripts/               # Generated transcripts
│   ├── summaries/                 # Generated summaries
│   └── vector_indices/            # FAISS vector indices
│
├── run_api.py                     # API server entry point
├── run_streamlit.py               # Streamlit app entry point
├── .env                           # Environment variables (API keys)
├── .env.example                   # Environment variables template
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Database Configuration

By default, SQLite is used. To use PostgreSQL, set in your `.env` file:

```
DATABASE_URL=postgresql://user:password@localhost/youtube_summarizer
```

## Caching Configuration

To enable Redis caching, add to your `.env` file:

```
REDIS_URL=redis://localhost:6379/0
```

If Redis is not available, automatic fallback to in-memory caching will occur.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- [Groq](https://groq.com/) for providing the API for Whisper and LLM models
- [PyTubeFix](https://github.com/JuanBindez/pytubefix) for YouTube downloading functionality
- [LangChain](https://www.langchain.com/) for LLM integration
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the web API framework
- [Streamlit](https://streamlit.io/) for the web interface
