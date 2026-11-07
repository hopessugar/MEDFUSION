"""Unit tests for text chunking and structure preservation."""

from __future__ import annotations

import pytest

from src.models.schemas import Document, Passage
from src.preprocessing.chunker import (
    ChunkingConfig,
    SimpleChunker,
    StructureAwareChunker,
)


def test_structure_aware_chunker_sections() -> None:
    """Test splitting across structured markdown headers."""
    doc = Document(
        doc_id="clinical_doc_01",
        title="Cardiovascular Disease Management",
        text=(
            "# Introduction\n\n"
            "Cardiovascular disease is the leading cause of mortality worldwide. "
            "Early intervention is critical.\n\n"
            "## Risk Assessment\n\n"
            "Assess 10-year ASCVD risk using standardized pooled cohort equations. "
            "Evaluate lipid profiles and blood pressure.\n\n"
            "## Therapy\n\n"
            "Initiate high-intensity statin therapy for secondary prevention. "
            "Target LDL-C reduction greater than 50 percent."
        ),
        metadata={"category": "Cardiology"},
    )

    config = ChunkingConfig(chunk_size=30, overlap=5, min_chunk_size=5, preserve_sections=True)
    chunker = StructureAwareChunker(config=config)
    passages = chunker.chunk_document(doc)

    assert len(passages) >= 3
    section_titles = {p.section for p in passages}
    assert "Introduction" in section_titles
    assert "Risk Assessment" in section_titles
    assert "Therapy" in section_titles

    for idx, p in enumerate(passages):
        assert p.doc_id == "clinical_doc_01"
        assert p.passage_id == f"clinical_doc_01_chunk_{idx:04d}"
        assert p.chunk_index == idx
        assert "word_count" in p.metadata


def test_chunker_overlap_and_sentence_preservation() -> None:
    """Test chunk overlap and that chunk boundaries respect sentence terminals."""
    sentences = [
        "First sentence defines pathophysiology.",
        "Second sentence explains diagnostic criteria.",
        "Third sentence details lab investigations.",
        "Fourth sentence discusses pharmacotherapy.",
        "Fifth sentence monitors clinical outcomes.",
    ]
    long_text = " ".join(sentences)
    doc = Document(doc_id="doc_overlap", title="Overlap Test", text=long_text)

    config = ChunkingConfig(chunk_size=15, overlap=5, min_chunk_size=5, preserve_sentences=True)
    chunker = StructureAwareChunker(config=config)
    passages = chunker.chunk_document(doc)

    assert len(passages) >= 2
    # Verify that each passage ends with a valid sentence boundary punctuation
    for p in passages:
        assert p.text.endswith((".", "!", "?", ";"))


def test_chunker_empty_and_short_documents() -> None:
    """Test chunker behavior on empty and minimal text."""
    chunker = StructureAwareChunker()

    empty_doc = Document(doc_id="empty_doc", title="Empty", text="   \n\t  ")
    assert chunker.chunk_document(empty_doc) == []

    short_doc = Document(doc_id="short_doc", title="Short", text="Penicillin allergy is documented.")
    short_passages = chunker.chunk_document(short_doc)
    assert len(short_passages) == 1
    assert short_passages[0].passage_id == "short_doc_chunk_0000"
    assert "Penicillin" in short_passages[0].text


def test_simple_chunker() -> None:
    """Test baseline SimpleChunker behavior."""
    doc = Document(
        doc_id="simple_doc",
        title="Simple Test",
        text="Word " * 100,
    )
    simple = SimpleChunker(chunk_size=25, overlap=5)
    passages = simple.chunk_document(doc)

    assert len(passages) >= 4
    for p in passages:
        assert p.doc_id == "simple_doc"
        assert len(p.text.split()) <= 30


def test_chunker_invalid_overlap() -> None:
    """Test that overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError):
        StructureAwareChunker(config=ChunkingConfig(chunk_size=20, overlap=25))
