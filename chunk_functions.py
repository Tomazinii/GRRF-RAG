from typing import List, Dict, Any, Union
import logging
import json

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker


import pdfplumber

def get_text_pdfplumber(pdf_path):
        whole_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    whole_text += f"{text}\n\n"
        return whole_text

logger = logging.getLogger(__name__)

def sliding_window_chunker(
    text: str,
    chunk_size: int = 1024,
    overlap: int = 256,
) -> List[str]:
    """
        Create overlapping chunks with a sliding window using TokenTextSplitter.

        Args:
            text (str): The text to chunk
            chunk_size (int, optional): The size of each chunk in tokens
            overlap (int, optional): The number of overlapping tokens between chunks

        Returns:
            List[str]: List of text chunks
    """
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    chunks = splitter.split_text(text)

    return chunks

def recursive_chunker(
    text: str,
    max_chunk_size: int = 1024,
    chunk_overlap: int = 256,
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
) -> List[str]:
    """
        Divide text based on semantic boundaries (like "\n\n", "\n", ". ", " ", "", or sentences, paragraphs, ...) using RecursiveCharacterTextSplitter.

        Args:
            text (str): The text to chunk
            max_chunk_size (int, optional): Maximum chunk size in tokens

        Returns:
            List[str]: List of text chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )

    chunks = splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} semantic chunks")

    return chunks

def markdown_chunker(
        text: str,
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
        strip_headers: bool = True
    ):
    """
    Divide text into chunks based on specific headers.

    Args:
        text: Text to be split.
        headers_to_split_on: List of TUPLES containing (header level, associated key name).
        strip_headers: If True, removes headers from chunks, else the chunk will have the headers before chunk content.
    Returns:
        List of object with page_content and metadata w/ headers.
    """

    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=strip_headers
    )

    chunks = text_splitter.split_text(text)
    return chunks


def semantic_chunker(
        text: str,
        embedding_model,
        breakpoint_threshold_amount: float = None,
        threshold_type: str = "percentile"
    ):
    """
    Splits a text into semantic chunks using SemanticChunker.

    Args:
        text (str): The input text to be split.
        embedding_model: The embedding model to use (that have embed_documents and embed_query methods. Langchain Embedding Class https://python.langchain.com/api_reference/core/embeddings/langchain_core.embeddings.embeddings.Embeddings.html#langchain_core.embeddings.embeddings.Embeddings).
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of characters that overlap between chunks.
        model_name (str): The name of the Hugging Face model to use if no embedding_model is provided.
        breakpoint_threshold_amount(float): The threshold amount used to determine breakpoints.
        threshold_type (str): The type of threshold used to determine breakpoints. Options available in LangChain:
            - "percentile": Based on a specified percentile of sentence similarity.
            - "standard_deviation": Uses standard deviation to find breakpoints.
            - "interquartile": Identifies outliers based on the interquartile range (IQR).
            - "gradient": Detects abrupt shifts in sentence similarity.


    Returns:
        list: A list of text chunks after semantic segmentation.
    """
    if embedding_model is None:
        return "No embedding model provided"

    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )

    return text_splitter.split_text(text)

def extractive_summary(text: str, limit: int) -> str:
    if limit < 30:
        raise ValueError("Limit too short, use at least 30")

    part = limit // 3

    start = text[:part].strip()
    mid_start = max((len(text) // 2) - (part // 2), 0)
    middle = text[mid_start:mid_start + part].strip()
    end = text[-part:].strip()

    return f"{start} ... {middle} ... {end}"


def hierarchical_chunking(
        text: str,
        doc_id: str,
        levels: List[str],
        document_size_limit: int = 3500,
        section_size_limit: int = 2000,
        section_size_overlap: int = 200,
        paragraph_size_limit: int = 500,
        paragraph_size_overlap: int = 50,
        sentence_size_limit: int = 100,
        sentence_size_overlap: int = 10,
    ):
    """
        Create multi-level chunks using a combination of splitters.

        Args:
            text (str): The text to chunk
            levels (List[str], optional): Hierarchy levels to include

        Returns:
            List[str]: List of text chunks with hierarchy information
    """
    # if levels is None:
    #     levels = ["document", "paragraph"]

    chunks = []
    document = None
    sections = []
    paragraphs = []
    sentences = []


    if "document" in levels:
        # chunks.append(f"[DOCUMENT] {extractive_summary(text, document_size_limit)}...")
        document = extractive_summary(text, document_size_limit)

    if "section" in levels:
        section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=section_size_limit,
            chunk_overlap=section_size_overlap,
            separators=["\n#", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". "],
        )
        section_chunks = section_splitter.split_text(text)
        
        for i, chunk in enumerate(section_chunks):
            chunks.append(f"[SECTION {i+1}] {chunk}")
            sections.append({"content": chunk, "chunk_type":"hierarchical_section", "doc_id": doc_id})

    if "paragraph" in levels:
        paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=paragraph_size_limit,
            chunk_overlap=paragraph_size_overlap,
            separators=["\n\n", "\n", ". "]
        )
        paragraph_chunks = paragraph_splitter.split_text(text)
        for i, chunk in enumerate(paragraph_chunks):
            chunks.append(f"[PARAGRAPH {i+1}] {chunk}")
            paragraphs.append({"content": chunk, "chunk_type":"hierarchical_paragraph", "doc_id": doc_id})

    if "sentence" in levels:
        sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=sentence_size_limit,
            chunk_overlap=sentence_size_overlap,
            separators=[". ", "! ", "? ", "; "]
        )
        sentence_chunks = sentence_splitter.split_text(text)
        for i, chunk in enumerate(sentence_chunks):
            chunks.append(f"[SENTENCE {i+1}] {chunk}")
            sentences.append({"content": chunk, "chunk_type":"hierarchical_sentence", "doc_id": doc_id})

    logger.info(
        f"Created {len(chunks)} hierarchical chunks across {len(levels)} levels"
    )

    return {
        "document": document,
        "sections": sections,
        "paragraphs": paragraphs,
        "sentences": sentences,
        "chunks": chunks
    }