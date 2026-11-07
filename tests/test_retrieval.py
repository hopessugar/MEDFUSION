"""Unit tests for the MedFusion-RAFT retrieval engine."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.document_loader import DocumentLoader, DocumentLoaderConfig
from src.models.schemas import Document, Passage, RetrievedPassage
from src.preprocessing.chunker import (
    ChunkingConfig,
    SimpleChunker,
    StructureAwareChunker,
)
from src.retrieval.bm25_retriever import BM25Config, BM25Retriever
from src.retrieval.dense_retriever import DenseRetrieverConfig, MockDenseRetriever
from src.retrieval.evaluation import (
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.retrieval.fusion import FusionConfig, HybridFusion
from src.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from src.retrieval.reranker import MockReranker, RerankerConfig


def test_document_loader():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        loader = DocumentLoader()

        # 1. Plain text
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Patient presented with acute myocardial infarction.", encoding="utf-8")
        docs = loader.load(txt_file)
        assert len(docs) == 1
        assert "myocardial infarction" in docs[0].text
        assert docs[0].doc_id

        # 2. Markdown
        md_file = tmp_path / "sample.md"
        md_file.write_text("# Clinical Guide\n\nPrescribe beta-blockers for hypertension.", encoding="utf-8")
        docs = loader.load(md_file)
        assert len(docs) == 1
        assert docs[0].title == "Clinical Guide"
        assert "beta-blockers" in docs[0].text

        # 3. JSON
        json_file = tmp_path / "sample.json"
        json_file.write_text(json.dumps([
            {"doc_id": "doc_001", "title": "Diabetes Care", "text": "Metformin is first-line."},
            {"doc_id": "doc_002", "title": "Asthma Care", "text": "Albuterol inhaler prescribed."}
        ]), encoding="utf-8")
        docs = loader.load(json_file)
        assert len(docs) == 2
        assert docs[0].doc_id == "doc_001"

        # 4. JSONL
        jsonl_file = tmp_path / "sample.jsonl"
        jsonl_file.write_text(
            '{"doc_id": "j1", "text": "Record 1"}\n{"doc_id": "j2", "text": "Record 2"}\n',
            encoding="utf-8",
        )
        docs = loader.load(jsonl_file)
        assert len(docs) == 2

        # 5. CSV
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("id,title,text\nc1,Title 1,Content 1\nc2,Title 2,Content 2\n", encoding="utf-8")
        docs = loader.load(csv_file)
        assert len(docs) == 2
        assert docs[0].doc_id == "c1"

        # 6. Directory load
        all_docs = loader.load_directory(tmp_path)
        assert len(all_docs) >= 7


def test_chunkers():
    doc = Document(
        doc_id="test_doc",
        title="Hypertension Treatment",
        text=(
            "# Introduction\n\n"
            "Hypertension is a major cardiovascular risk factor worldwide. "
            "It affects over one billion people globally.\n\n"
            "## Management\n\n"
            "Lifestyle interventions include sodium restriction and exercise. "
            "First-line pharmacotherapy includes ACE inhibitors, ARBs, and calcium channel blockers. "
            "Beta-blockers may be considered in patients with ischemic heart disease."
        ),
    )

    # Structure-aware chunker
    chunker = StructureAwareChunker(ChunkingConfig(chunk_size=30, overlap=10, min_chunk_size=5))
    passages = chunker.chunk_document(doc)
    assert len(passages) >= 2
    for p in passages:
        assert p.doc_id == "test_doc"
        assert p.passage_id.startswith("test_doc_chunk_")
        assert p.section in ["Introduction", "Management"]
        assert "word_count" in p.metadata

    # Simple chunker
    simple = SimpleChunker(chunk_size=20, overlap=5)
    simple_passages = simple.chunk_document(doc)
    assert len(simple_passages) >= 2


def test_bm25_retriever():
    passages = [
        Passage(passage_id="p1", doc_id="d1", text="Type 2 diabetes mellitus management with metformin."),
        Passage(passage_id="p2", doc_id="d2", text="Acute asthma exacerbation treated with albuterol and steroids."),
        Passage(passage_id="p3", doc_id="d3", text="Cardiovascular risk reduction using statins and aspirin."),
    ]

    retriever = BM25Retriever(BM25Config(top_k=2))
    retriever.build_index(passages)

    results = retriever.search("asthma exacerbation", top_k=2)
    assert len(results) == 2
    assert results[0].passage_id == "p2"
    assert results[0].retrieval_method == "bm25"
    assert results[0].score > 0.0

    # Test serialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_path = Path(tmp_dir) / "bm25.pkl"
        retriever.save_index(idx_path)
        loaded = BM25Retriever()
        loaded.load_index(idx_path)
        loaded_res = loaded.search("asthma exacerbation", top_k=2)
        assert loaded_res[0].passage_id == "p2"


def test_mock_dense_retriever():
    passages = [
        Passage(passage_id="p1", doc_id="d1", text="Chemotherapy protocols for breast cancer."),
        Passage(passage_id="p2", doc_id="d2", text="Radiation therapy targeting glioblastoma."),
    ]
    retriever = MockDenseRetriever(DenseRetrieverConfig(embedding_dim=128, top_k=2))
    retriever.build_index(passages)

    results = retriever.search("radiation glioblastoma", top_k=2)
    assert len(results) == 2
    assert results[0].retrieval_method == "dense_medcpt"

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "dense"
        retriever.save_index(save_path)
        loaded = MockDenseRetriever(DenseRetrieverConfig(embedding_dim=128))
        loaded.load_index(save_path)
        assert len(loaded.passages) == 2


def test_hybrid_fusion():
    bm25_res = [
        RetrievedPassage(passage_id="p1", doc_id="d1", text="Text 1", score=10.0, retrieval_method="bm25", rank=1),
        RetrievedPassage(passage_id="p2", doc_id="d2", text="Text 2", score=5.0, retrieval_method="bm25", rank=2),
    ]
    dense_res = [
        RetrievedPassage(passage_id="p2", doc_id="d2", text="Text 2", score=0.9, retrieval_method="dense", rank=1),
        RetrievedPassage(passage_id="p3", doc_id="d3", text="Text 3", score=0.7, retrieval_method="dense", rank=2),
    ]

    fusion = HybridFusion(FusionConfig(rrf_k=60, top_k=3))
    fused = fusion.fuse([bm25_res, dense_res], weights=[1.0, 1.0])

    assert len(fused) == 3
    # p2 was present in both lists, so its reciprocal rank sum should place it at or near the top
    assert fused[0].passage_id == "p2"
    assert fused[0].retrieval_method == "hybrid_rrf"
    assert "fusion_details" in fused[0].metadata


def test_mock_reranker():
    candidates = [
        RetrievedPassage(passage_id="p1", doc_id="d1", text="General medical overview", score=0.5, retrieval_method="bm25"),
        RetrievedPassage(passage_id="p2", doc_id="d2", text="Cardiovascular infarction treatment", score=0.4, retrieval_method="bm25"),
    ]
    reranker = MockReranker(RerankerConfig(top_k=2))
    reranked = reranker.rerank("cardiovascular treatment", candidates)
    assert len(reranked) == 2
    assert reranked[0].passage_id == "p2"
    assert reranked[0].retrieval_method.endswith("_reranked")


def test_retrieval_pipeline_end_to_end():
    passages = [
        Passage(passage_id="p1", doc_id="d1", text="Treatment of type 2 diabetes with metformin."),
        Passage(passage_id="p2", doc_id="d2", text="Inhaled corticosteroids for persistent asthma."),
        Passage(passage_id="p3", doc_id="d3", text="Hypertension managed with ACE inhibitors."),
    ]
    cfg = RetrievalPipelineConfig(
        use_mock_components=True,
        bm25_top_k=3,
        dense_top_k=3,
        final_top_k=2,
    )
    pipeline = RetrievalPipeline(config=cfg)
    pipeline.build_index(passages)

    results = pipeline.retrieve("asthma inhalers", top_k=2)
    assert len(results) <= 2
    assert "pipeline_timings_ms" in results[0].metadata

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_indices(tmp_dir)
        new_pipeline = RetrievalPipeline(config=cfg)
        new_pipeline.load_indices(tmp_dir)
        loaded_res = new_pipeline.retrieve("diabetes metformin", top_k=1)
        assert len(loaded_res) == 1


def test_evaluation_metrics():
    retrieved = ["p1", "p2", "p3", "p4", "p5"]
    relevant = {"p2", "p5"}

    # Recall@K
    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=5) == 1.0

    # Precision@K
    assert precision_at_k(retrieved, relevant, k=1) == 0.0
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=5) == 0.4

    # MRR
    assert mrr(retrieved, relevant) == 0.5  # First hit is p2 at rank 2

    # NDCG@K
    assert ndcg_at_k(retrieved, relevant, k=2) > 0.0

    # evaluate_retrieval
    queries = ["q1"]
    ret_passages = [[RetrievedPassage(passage_id=pid, doc_id="d", text="t", score=1.0) for pid in retrieved]]
    metrics = evaluate_retrieval(queries, ret_passages, [relevant], k_values=[1, 2, 5])
    assert metrics["num_queries"] == 1.0
    assert metrics["MRR"] == 0.5
    assert metrics["Recall@2"] == 0.5
