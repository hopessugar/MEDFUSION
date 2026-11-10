"""Text chunking module for MedFusion-RAFT.

Provides structure-aware and simple fixed-size text chunking strategies that output
validated Passage schema objects with preserved section and document metadata.
"""

import abc
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.models.schemas import Document, Passage

logger = logging.getLogger(__name__)


class ChunkingConfig(BaseModel):
    """Configuration options for document chunking."""

    chunk_size: int = Field(
        default=256,
        ge=10,
        description="Target maximum size of each chunk in words",
    )
    overlap: int = Field(
        default=40,
        ge=0,
        description="Number of words to overlap between consecutive chunks",
    )
    min_chunk_size: int = Field(
        default=30,
        ge=5,
        description="Minimum word count required for a standalone chunk",
    )
    preserve_sections: bool = Field(
        default=True,
        description="Prevent chunks from crossing major section boundaries",
    )
    preserve_sentences: bool = Field(
        default=True,
        description="Ensure chunk boundaries only occur at sentence boundaries",
    )


class BaseChunker(abc.ABC):
    """Abstract base class for all document chunking implementations."""

    @abc.abstractmethod
    def chunk_document(self, document: Document) -> List[Passage]:
        """Split a single Document into a list of Passage chunks.

        Args:
            document: Document object to chunk.

        Returns:
            List of generated Passage objects.
        """
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Passage]:
        """Split a collection of documents into Passages.

        Args:
            documents: List of Document objects to chunk.

        Returns:
            Flat list of Passage objects across all documents.
        """
        all_passages: List[Passage] = []
        for doc in documents:
            passages = self.chunk_document(doc)
            all_passages.extend(passages)
        logger.info(
            "Chunked %d documents into %d total passages",
            len(documents),
            len(all_passages),
        )
        return all_passages


class StructureAwareChunker(BaseChunker):
    """Structure-aware chunker that respects headers, paragraphs, and sentences.

    Splits text hierarchically:
        1. Sections (Markdown headers, numbered sections like '1. Introduction')
        2. Paragraphs within sections
        3. Sentences within paragraphs (with medical abbreviation awareness)
    """

    # Matches sentence boundary candidates
    SENTENCE_SPLIT_PATTERN = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])",
        re.IGNORECASE,
    )
    COMMON_ABBREVIATIONS = (
        "dr.", "prof.", "mr.", "mrs.", "ms.", "vs.", "eg.", "ie.",
        "al.", "fig.", "tab.", "ref.", "no.", "approx.", "e.g.", "i.e.",
    )

    # Matches markdown headers (# Header) or numbered headings (1. Intro, 1.1 Methods, Section 1:)
    HEADER_PATTERN = re.compile(
        r"^(?:#{1,6}\s+.+|(?:\d+\.)+\d*\s+[A-Z].+|Section\s+\d+[:\s].+)$",
        re.MULTILINE,
    )

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        """Initialize StructureAwareChunker.

        Args:
            config: Chunking configuration parameters.
        """
        self.config = config or ChunkingConfig()
        if self.config.overlap >= self.config.chunk_size:
            raise ValueError(
                f"Overlap ({self.config.overlap}) must be smaller than chunk_size ({self.config.chunk_size})"
            )

    def chunk_document(self, document: Document) -> List[Passage]:
        """Split a document into passages respecting structure.

        Args:
            document: Source document to split.

        Returns:
            List of Passage instances with stable IDs and metadata.
        """
        # Handle page-structured documents (PDFs) differently
        if document.pages and len(document.pages) > 0:
            return self._chunk_pages(document)
        
        text = document.text.strip()
        if not text:
            return []

        sections = self._split_into_sections(text) if self.config.preserve_sections else [("General", text)]
        passages: List[Passage] = []
        chunk_idx = 0

        for section_title, section_text in sections:
            section_passages = self._chunk_section(
                doc=document,
                section_title=section_title,
                section_text=section_text,
                start_index=chunk_idx,
            )
            passages.extend(section_passages)
            chunk_idx += len(section_passages)

        # Fallback if no passages were created (e.g., text was very short but non-empty)
        if not passages and text:
            passages.append(
                Passage(
                    passage_id=f"{document.doc_id}_chunk_0000",
                    doc_id=document.doc_id,
                    text=text,
                    section="General",
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "chunk_index": 0,
                        "word_count": len(text.split()),
                    },
                )
            )

        return passages
    
    def _chunk_pages(self, document: Document) -> List[Passage]:
        """Chunk a page-structured document (like PDF) preserving page boundaries.
        
        Args:
            document: Document with page-level structure
            
        Returns:
            List of Passage instances with page metadata preserved
        """
        passages: List[Passage] = []
        chunk_idx = 0
        
        for page in document.pages:
            page_text = page.text.strip()
            if not page_text:
                continue
            
            # Get section from page if available, or detect from content
            page_sections = self._split_into_sections(page_text) if self.config.preserve_sections else [("General", page_text)]
            
            for section_title, section_text in page_sections:
                section_passages = self._chunk_section(
                    doc=document,
                    section_title=section_title,
                    section_text=section_text,
                    start_index=chunk_idx,
                    page_number=page.page_number,
                    page_metadata=page.metadata,
                )
                passages.extend(section_passages)
                chunk_idx += len(section_passages)
        
        return passages

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text into (section_header, section_body) pairs."""
        matches = list(self.HEADER_PATTERN.finditer(text))
        if not matches:
            return [("Document Content", text)]

        sections: List[Tuple[str, str]] = []
        prev_header = "Introduction"
        prev_end = 0

        for match in matches:
            start, end = match.span()
            header_line = match.group(0).strip().lstrip("#").strip()
            # Text before the first header (if any)
            body_before = text[prev_end:start].strip()
            if body_before:
                sections.append((prev_header, body_before))
            prev_header = header_line
            prev_end = end

        # Trailing section after last header
        trailing = text[prev_end:].strip()
        if trailing:
            sections.append((prev_header, trailing))

        return sections if sections else [("Document Content", text)]

    def _split_into_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into individual sentences using medical-aware regex."""
        raw_splits = self.SENTENCE_SPLIT_PATTERN.split(paragraph.strip())
        sentences: List[str] = []
        for part in raw_splits:
            part_s = part.strip()
            if not part_s:
                continue
            if sentences and any(sentences[-1].lower().endswith(abbr) for abbr in self.COMMON_ABBREVIATIONS):
                sentences[-1] = f"{sentences[-1]} {part_s}"
            else:
                sentences.append(part_s)
        return sentences

    def _chunk_section(
        self,
        doc: Document,
        section_title: str,
        section_text: str,
        start_index: int,
        page_number: Optional[int] = None,
        page_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Passage]:
        """Chunk sentences within a section into sized passages.
        
        Args:
            doc: Source document
            section_title: Title of the section being chunked
            section_text: Text content of the section
            start_index: Starting chunk index
            page_number: Optional page number for PDF documents
            page_metadata: Optional page-level metadata
        """
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        # Collect all sentences across paragraphs
        all_sentences: List[str] = []
        for p in paragraphs:
            sents = self._split_into_sentences(p)
            if sents:
                all_sentences.extend(sents)

        if not all_sentences:
            return []

        passages: List[Passage] = []
        current_sentences: List[str] = []
        current_words = 0
        chunk_idx = start_index

        i = 0
        while i < len(all_sentences):
            sentence = all_sentences[i]
            sentence_words = len(sentence.split())

            # If a single sentence is larger than chunk_size, split by words
            if sentence_words > self.config.chunk_size and not self.config.preserve_sentences:
                words = sentence.split()
                w_start = 0
                while w_start < len(words):
                    w_end = min(w_start + self.config.chunk_size, len(words))
                    sub_text = " ".join(words[w_start:w_end])
                    passages.append(
                        self._create_passage(
                            doc=doc,
                            text=sub_text,
                            section=section_title,
                            chunk_index=chunk_idx,
                            page_number=page_number,
                            page_metadata=page_metadata,
                        )
                    )
                    chunk_idx += 1
                    w_start += max(1, self.config.chunk_size - self.config.overlap)
                i += 1
                continue

            # Check if adding this sentence exceeds chunk_size
            if current_sentences and (current_words + sentence_words > self.config.chunk_size):
                chunk_text = " ".join(current_sentences)
                if current_words >= self.config.min_chunk_size:
                    passages.append(
                        self._create_passage(
                            doc=doc,
                            text=chunk_text,
                            section=section_title,
                            chunk_index=chunk_idx,
                            page_number=page_number,
                            page_metadata=page_metadata,
                        )
                    )
                    chunk_idx += 1

                # Handle overlap: backtrack sentences to keep overlap words
                overlap_words = 0
                overlap_sentences: List[str] = []
                for s in reversed(current_sentences):
                    cnt = len(s.split())
                    if overlap_words + cnt <= self.config.overlap or not overlap_sentences:
                        overlap_sentences.insert(0, s)
                        overlap_words += cnt
                    else:
                        break

                current_sentences = overlap_sentences
                current_words = overlap_words

            current_sentences.append(sentence)
            current_words += sentence_words
            i += 1

        # Flush any remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            # If remaining is smaller than min_chunk_size and we have previous chunk in this section, merge
            if current_words < self.config.min_chunk_size and passages:
                last_passage = passages[-1]
                merged_text = f"{last_passage.text} {chunk_text}"
                passages[-1] = self._create_passage(
                    doc=doc,
                    text=merged_text,
                    section=section_title,
                    chunk_index=last_passage.metadata["chunk_index"],
                    page_number=page_number,
                    page_metadata=page_metadata,
                )
            else:
                passages.append(
                    self._create_passage(
                        doc=doc,
                        text=chunk_text,
                        section=section_title,
                        chunk_index=chunk_idx,
                        page_number=page_number,
                        page_metadata=page_metadata,
                    )
                )

        return passages

    def _create_passage(
        self,
        doc: Document,
        text: str,
        section: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        page_metadata: Optional[Dict[str, Any]] = None,
    ) -> Passage:
        """Create a Passage object with consistent metadata and identifier."""
        passage_id = f"{doc.doc_id}_chunk_{chunk_index:04d}"
        metadata: Dict[str, Any] = {
            **doc.metadata,
            "title": doc.title,
            "section": section,
            "chunk_index": chunk_index,
            "word_count": len(text.split()),
            "char_count": len(text),
        }
        
        # Add page information if available
        if page_number is not None:
            metadata["page_number"] = page_number
        if page_metadata:
            metadata["page_metadata"] = page_metadata
        
        return Passage(
            passage_id=passage_id,
            doc_id=doc.doc_id,
            text=text,
            metadata=metadata,
            section=section,
            chunk_index=chunk_index,
        )


class SimpleChunker(BaseChunker):
    """Simpler fixed-size sliding window chunker fallback."""

    def __init__(self, chunk_size: int = 256, overlap: int = 40) -> None:
        """Initialize SimpleChunker.

        Args:
            chunk_size: Window size in words.
            overlap: Overlap in words.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

    def chunk_document(self, document: Document) -> List[Passage]:
        """Split document text into fixed word windows."""
        words = document.text.split()
        if not words:
            return []

        passages: List[Passage] = []
        chunk_idx = 0
        step = max(1, self.chunk_size - self.overlap)

        for start in range(0, len(words), step):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            passage_id = f"{document.doc_id}_chunk_{chunk_idx:04d}"
            passages.append(
                Passage(
                    passage_id=passage_id,
                    doc_id=document.doc_id,
                    text=chunk_text,
                    section="General",
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "chunk_index": chunk_idx,
                        "word_count": len(chunk_words),
                    },
                )
            )
            chunk_idx += 1

            if end >= len(words):
                break

        return passages
