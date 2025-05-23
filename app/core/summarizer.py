"""
Module for summarizing transcripts using LLM models.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from app.models.schemas import YouTubeMedia, SummaryConfig, VideoSummary, TranscriptedData
from app.core.prompts import SUMMARIZE_SYSTEM_PROMPT, REDUCE_PROMPOT

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
        document = Document(page_content=transcript_text)

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

        system_prompt = SUMMARIZE_SYSTEM_PROMPT

        if config.num_lines:
            system_prompt += f" in exactly {config.num_lines} lines"

        if config.selective_keywords:
            system_prompt += f". Focus on these keywords: {config.selective_keywords}"

        system_prompt += ":\n\n{text}"

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])

        # For shorter transcripts: use the "stuff" method
        if len(docs) == 1:
            chain = summary_prompt | llm
            summary = chain.invoke({"text": transcript_text})
            return summary.content

        # For longer transcripts: use map-reduce
        else:
            # First summarize each chunk
            # Construct dynamic prompt for chunk summarization
            map_system_prompt = "Summarize this part of a transcript"

            if config.selective_keywords:
                map_system_prompt += f", focusing on these keywords: {config.selective_keywords}"

            map_system_prompt += ":\n\n{text}"

            map_prompt = ChatPromptTemplate.from_messages([
                ("system", map_system_prompt)
            ])
            map_chain = map_prompt | llm

            interim_summaries = []
            for doc in docs:
                interim_summary = map_chain.invoke({"text": doc.page_content})
                interim_summaries.append(interim_summary.content)

            # Then combine the summaries
            # Construct dynamic prompt for final summarization
            reduce_system_prompt = REDUCE_PROMPOT

            if config.num_lines:
                reduce_system_prompt += f" in exactly {config.num_lines} lines"

            if config.selective_keywords:
                reduce_system_prompt += f", focusing on these keywords: {config.selective_keywords}"

            reduce_system_prompt += ":\n\n{summaries}"

            reduce_prompt = ChatPromptTemplate.from_messages([
                ("system", reduce_system_prompt)
            ])
            reduce_chain = reduce_prompt | llm

            final_summary = reduce_chain.invoke({"summaries": "\n\n".join(interim_summaries)})
            return final_summary.content

    def create_summary(self, media: YouTubeMedia, transcript_data:TranscriptedData , config: SummaryConfig) -> VideoSummary:
        """
        Create a full video summary.

        Args:
            media: YouTubeMedia object
            transcript_data : TranscriptedData object with transcript text and segments
            config: Configuration for summarization

        Returns:
            VideoSummary object
        """
        summary = self.summarize(transcript_data.transcript_text, config)

        return VideoSummary(
            media_info=media,
            summary=summary,
            transcript_text=transcript_data.transcript_text,
            transcript_segments=transcript_data.segments,
            language=transcript_data.language,
            model=transcript_data.model
        )