"""
Module for summarizing transcripts using LLM models.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from app.models.schemas import YouTubeMedia, SummaryConfig, VideoSummary


class TranscriptSummarizer:
    """Class to handle transcript summarization operations."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the summarizer with API key.

        Args:
            api_key: Groq API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set it in .env file or pass directly.")

        os.environ["GROQ_API_KEY"] = self.api_key

    def summarize(self, transcript_text: str, config: SummaryConfig) -> str:
        """
        Summarize a transcript text.

        Args:
            transcript_text: Full transcript text to summarize
            config: Configuration for summarization

        Returns:
            Summarized text
        """
        # Create a Document object
        document = Document(page_content=transcript_text)

        # For longer transcripts, split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        docs = text_splitter.split_documents([document])

        llm = init_chat_model(
            model=config.model,
            model_provider="groq",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Create a concise summary of the following transcript from a YouTube video:\n\n{text}")
        ])

        # For shorter transcripts: use the "stuff" method
        if len(docs) == 1:
            chain = summary_prompt | llm
            summary = chain.invoke({"text": transcript_text})
            return summary.content

        # For longer transcripts: use map-reduce
        else:
            # First summarize each chunk
            map_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize this part of a transcript:\n\n{text}")
            ])
            map_chain = map_prompt | llm

            interim_summaries = []
            for doc in docs:
                interim_summary = map_chain.invoke({"text": doc.page_content})
                interim_summaries.append(interim_summary.content)

            # Then combine the summaries
            reduce_prompt = ChatPromptTemplate.from_messages([
                ("system", "Combine these partial summaries into a coherent overall summary:\n\n{summaries}")
            ])
            reduce_chain = reduce_prompt | llm

            final_summary = reduce_chain.invoke({"summaries": "\n\n".join(interim_summaries)})
            return final_summary.content

    def create_summary(self, media: YouTubeMedia, transcript_text: str, config: SummaryConfig) -> VideoSummary:
        """
        Create a full video summary.

        Args:
            media: YouTubeMedia object
            transcript_text: Full transcript text
            config: Configuration for summarization

        Returns:
            VideoSummary object
        """
        summary = self.summarize(transcript_text, config)

        return VideoSummary(
            media_info=media,
            summary=summary,
            transcript_text=transcript_text
        )