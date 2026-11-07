"""Integration tests for FastAPI endpoints using TestClient."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.services.pipeline_service import PipelineService


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a persistent FastAPI TestClient instance configured in mock mode."""
    PipelineService.reset_instance()
    app = create_app(mock=True)
    with TestClient(app) as test_client:
        yield test_client
    PipelineService.reset_instance()


def test_health_endpoint(client: TestClient) -> None:
    """Test GET /api/v1/health returns 200 with status fields."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert "uptime_seconds" in data
    assert "version" in data


def test_ask_endpoint(client: TestClient) -> None:
    """Test POST /api/v1/ask processes a valid clinical question."""
    payload = {
        "question": "What is the recommended first-line therapy for uncomplicated hypertension?",
        "specialty": "Cardiology",
        "use_retrieval": True,
        "use_verification": True,
        "use_abstention": True,
        "experiment_id": "E5_raft",
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "answer_text" in data
    assert len(data["answer_text"]) > 0
    assert data["answerability"] == "answerable"
    assert data["decision"] in ("accept", "revise", "abstain")
    assert isinstance(data["claims"], list)
    assert isinstance(data["evidence_passages"], list)
    assert data["processing_time_ms"] >= 0.0


def test_batch_endpoint(client: TestClient) -> None:
    """Test POST /api/v1/batch processes multiple clinical questions."""
    payload = {
        "questions": [
            {"question": "How is asthma diagnosed in primary care?"},
            {"question": "What is the role of metformin in type 2 diabetes?"},
        ],
        "experiment_id": "E9_full",
    }
    response = client.post("/api/v1/batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert len(data["results"]) == 2
    assert data["total_time_ms"] >= 0.0
    assert data["experiment_id"] == "E9_full"


def test_config_endpoint(client: TestClient) -> None:
    """Test GET /api/v1/config returns active configuration dictionary."""
    response = client.get("/api/v1/config")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert "retrieval" in data
    assert "experiments" in data


def test_evaluate_endpoint(client: TestClient) -> None:
    """Test POST /api/v1/evaluate executes benchmark evaluation over a dataset file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_file = Path(tmp_dir) / "test_eval_data.json"
        sample_dataset = [
            {"question": "What treats diabetic ketoacidosis?", "specialty": "Endocrinology"},
            {"question": "What is the initial management of STEMI?", "specialty": "Cardiology"},
        ]
        dataset_file.write_text(json.dumps(sample_dataset), encoding="utf-8")

        payload = {
            "dataset_path": str(dataset_file),
            "experiment_id": "E5_raft",
            "output_dir": str(Path(tmp_dir) / "eval_out"),
        }
        response = client.post("/api/v1/evaluate", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["num_evaluated"] == 2
        assert data["experiment_id"] == "E5_raft"
        assert "decisions" in data
        assert Path(data["output_file"]).is_file()


def test_request_validation_errors(client: TestClient) -> None:
    """Test 422 error handling on missing fields, forbidden extra fields, and invalid constraints."""
    # 1. Question too short (< 3 chars)
    resp1 = client.post("/api/v1/ask", json={"question": "No"})
    assert resp1.status_code == 422
    assert "Validation Error" in resp1.json()["error"]

    # 2. Forbidden extra fields
    resp2 = client.post("/api/v1/ask", json={"question": "Valid question?", "unauthorized": "payload"})
    assert resp2.status_code == 422

    # 3. Batch with empty questions list
    resp3 = client.post("/api/v1/batch", json={"questions": []})
    assert resp3.status_code == 422


def test_evaluate_missing_file_error(client: TestClient) -> None:
    """Test 404 error returned when dataset file does not exist."""
    payload = {
        "dataset_path": "non_existent_path_file_9876.json",
        "experiment_id": "E1_base",
    }
    response = client.post("/api/v1/evaluate", json=payload)
    assert response.status_code == 404
    assert "Dataset file not found" in response.json()["error"]
