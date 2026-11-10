#!/usr/bin/env python3
"""Dataset preparation script for MedFusion-RAFT.

Prepares SFT and RAFT training datasets from public medical QA benchmarks
and custom medical document collections.

Usage:
    python scripts/prepare_data.py --output-dir data/processed
    python scripts/prepare_data.py --output-dir data/processed --include-pubmedqa
    python scripts/prepare_data.py --output-dir data/processed --build-raft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import (
    AnswerabilityLabel,
    DatasetExample,
    Passage,
    TrainingRAFTExample,
    TrainingSFTExample,
)


def load_pubmedqa() -> list[DatasetExample]:
    """Load PubMedQA dataset from HuggingFace."""
    examples = []
    try:
        from datasets import load_dataset
        print("📥 Loading PubMedQA from HuggingFace...")
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)

        for i, item in enumerate(dataset):
            question = item.get("question", "")
            context_list = item.get("context", {}).get("contexts", [])
            context = " ".join(context_list) if isinstance(context_list, list) else str(context_list)
            answer = item.get("long_answer", "")
            final_decision = item.get("final_decision", "")

            answerability = AnswerabilityLabel.ANSWERABLE
            if final_decision == "maybe":
                answerability = AnswerabilityLabel.PARTIALLY_ANSWERABLE

            example = DatasetExample(
                question_id=f"pubmedqa_{i:05d}",
                question=question,
                reference_answer=answer,
                gold_passage=context[:1000] if context else None,
                gold_passage_id=f"pubmedqa_{i:05d}_p0",
                doc_id=f"pubmedqa_{item.get('pubid', i)}",
                specialty="general_medicine",
                difficulty=None,
                answerability=answerability,
                source_url="https://huggingface.co/datasets/qiaojin/PubMedQA",
                review_status="benchmark",
            )
            examples.append(example)

        print(f"   Loaded {len(examples)} PubMedQA examples")
    except ImportError:
        print("⚠️  `datasets` library not installed. Skipping PubMedQA.")
    except Exception as e:
        print(f"⚠️  Failed to load PubMedQA: {e}")

    return examples


def create_sample_medical_dataset() -> list[DatasetExample]:
    """Create a small sample medical QA dataset for testing."""
    print("📝 Creating sample medical QA dataset...")
    samples = [
        DatasetExample(
            question_id="sample_001",
            question="What is the first-line treatment for type 2 diabetes mellitus?",
            reference_answer="Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus, combined with lifestyle modifications including diet and exercise.",
            gold_passage="Metformin is recommended as the initial pharmacological agent for the treatment of type 2 diabetes. It decreases hepatic glucose production and increases insulin sensitivity. Lifestyle modifications including dietary changes and regular physical activity should accompany pharmacological therapy.",
            gold_passage_id="sample_001_p0",
            doc_id="diabetes_guidelines_2023",
            specialty="endocrinology",
            difficulty="easy",
            answerability=AnswerabilityLabel.ANSWERABLE,
            review_status="verified",
        ),
        DatasetExample(
            question_id="sample_002",
            question="What are the contraindications for metformin use?",
            reference_answer="Metformin is contraindicated in patients with severe renal impairment (eGFR < 30), acute or chronic metabolic acidosis, and in situations with risk of lactic acidosis such as sepsis or acute heart failure.",
            gold_passage="Contraindications for metformin include: severe renal impairment (eGFR below 30 mL/min/1.73m2), acute or chronic metabolic acidosis including diabetic ketoacidosis, and conditions that may predispose to lactic acidosis such as sepsis, dehydration, excess alcohol intake, hepatic insufficiency, or acute heart failure.",
            gold_passage_id="sample_002_p0",
            doc_id="diabetes_guidelines_2023",
            specialty="endocrinology",
            difficulty="medium",
            answerability=AnswerabilityLabel.ANSWERABLE,
            review_status="verified",
        ),
        DatasetExample(
            question_id="sample_003",
            question="What is the recommended blood pressure target for patients with diabetes?",
            reference_answer="For most adults with diabetes, a blood pressure target of less than 130/80 mmHg is recommended, though targets may be individualized based on patient factors.",
            gold_passage="The American Diabetes Association recommends a blood pressure target of less than 130/80 mmHg for most adults with diabetes. Individualization of targets may be appropriate based on patient characteristics including age, cardiovascular risk, and tolerance of medications.",
            gold_passage_id="sample_003_p0",
            doc_id="hypertension_diabetes_2023",
            specialty="cardiology",
            difficulty="easy",
            answerability=AnswerabilityLabel.ANSWERABLE,
            review_status="verified",
        ),
        DatasetExample(
            question_id="sample_004",
            question="What is the cure rate for stage IV pancreatic cancer?",
            reference_answer="The evidence does not provide specific cure rate data for stage IV pancreatic cancer.",
            gold_passage="Pancreatic cancer remains one of the most lethal malignancies. Early-stage detection through screening has shown potential for improving outcomes, but the majority of cases are diagnosed at advanced stages.",
            gold_passage_id="sample_004_p0",
            doc_id="oncology_review_2023",
            specialty="oncology",
            difficulty="hard",
            answerability=AnswerabilityLabel.PARTIALLY_ANSWERABLE,
            review_status="verified",
        ),
        DatasetExample(
            question_id="sample_005",
            question="What is the quantum mechanical basis for protein folding in Alzheimer's disease?",
            reference_answer="The provided evidence does not contain information about quantum mechanical aspects of protein folding in Alzheimer's disease.",
            gold_passage="Alzheimer's disease is characterized by the accumulation of amyloid-beta plaques and neurofibrillary tangles. Current therapeutic approaches include cholinesterase inhibitors and NMDA receptor antagonists.",
            gold_passage_id="sample_005_p0",
            doc_id="neurology_review_2023",
            specialty="neurology",
            difficulty="hard",
            answerability=AnswerabilityLabel.UNANSWERABLE,
            review_status="verified",
        ),
    ]

    # Add more diverse samples
    for i in range(6, 16):
        samples.append(DatasetExample(
            question_id=f"sample_{i:03d}",
            question=f"Sample medical question {i} about common conditions?",
            reference_answer=f"Sample reference answer {i} with medical content.",
            gold_passage=f"Sample evidence passage {i} with relevant medical information for the question.",
            gold_passage_id=f"sample_{i:03d}_p0",
            doc_id=f"medical_doc_{(i % 3) + 1}",
            specialty=["cardiology", "endocrinology", "neurology", "oncology", "general_medicine"][i % 5],
            difficulty=["easy", "medium", "hard"][i % 3],
            answerability=[AnswerabilityLabel.ANSWERABLE, AnswerabilityLabel.ANSWERABLE, AnswerabilityLabel.PARTIALLY_ANSWERABLE][i % 3],
            review_status="sample",
        ))

    print(f"   Created {len(samples)} sample examples")
    return samples


def split_by_document(
    examples: list[DatasetExample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[DatasetExample]]:
    """Split dataset by source document to prevent leakage."""
    random.seed(seed)

    # Group by doc_id
    doc_groups: dict[str, list[DatasetExample]] = defaultdict(list)
    for ex in examples:
        doc_id = ex.doc_id or f"ungrouped_{hashlib.md5(ex.question.encode()).hexdigest()[:8]}"
        doc_groups[doc_id].append(ex)

    # Shuffle document IDs
    doc_ids = list(doc_groups.keys())
    random.shuffle(doc_ids)

    # Split documents
    n_docs = len(doc_ids)
    n_train = max(1, int(n_docs * train_ratio))
    n_val = max(1, int(n_docs * val_ratio))

    train_docs = doc_ids[:n_train]
    val_docs = doc_ids[n_train:n_train + n_val]
    test_docs = doc_ids[n_train + n_val:]

    splits = {
        "train": [ex for doc_id in train_docs for ex in doc_groups[doc_id]],
        "validation": [ex for doc_id in val_docs for ex in doc_groups[doc_id]],
        "test": [ex for doc_id in test_docs for ex in doc_groups[doc_id]],
    }

    print(f"   Split by document: {len(train_docs)} train docs, {len(val_docs)} val docs, {len(test_docs)} test docs")
    print(f"   Examples: {len(splits['train'])} train, {len(splits['validation'])} val, {len(splits['test'])} test")

    return splits


def normalize_question(question: str) -> str:
    """Normalize question for similarity comparison."""
    # Convert to lowercase
    q = question.lower()
    # Remove punctuation
    q = re.sub(r'[^\w\s]', '', q)
    # Remove extra whitespace
    q = ' '.join(q.split())
    return q


def compute_jaccard_similarity(str1: str, str2: str) -> float:
    """Compute Jaccard similarity between two strings (word-level)."""
    words1 = set(str1.split())
    words2 = set(str2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0


def detect_near_duplicates(
    splits: dict[str, list[DatasetExample]],
    similarity_threshold: float = 0.85
) -> dict:
    """Detect near-duplicate questions across train/val/test splits.
    
    Uses Jaccard similarity on normalized questions to find potential leakage.
    
    Args:
        splits: Dictionary with 'train', 'validation', 'test' splits
        similarity_threshold: Threshold for flagging as near-duplicate (0.0-1.0)
    
    Returns:
        Dictionary with contamination report
    """
    print(f"\n🔍 Detecting near-duplicates (threshold={similarity_threshold})...")
    
    # Normalize all questions
    train_qs = [(i, normalize_question(ex.question)) for i, ex in enumerate(splits['train'])]
    val_qs = [(i, normalize_question(ex.question)) for i, ex in enumerate(splits['validation'])]
    test_qs = [(i, normalize_question(ex.question)) for i, ex in enumerate(splits['test'])]
    
    report = {
        "train_val_duplicates": [],
        "train_test_duplicates": [],
        "val_test_duplicates": [],
        "total_flags": 0
    }
    
    # Check train vs validation
    for test_idx, test_q in val_qs:
        for train_idx, train_q in train_qs:
            sim = compute_jaccard_similarity(test_q, train_q)
            if sim >= similarity_threshold:
                report["train_val_duplicates"].append({
                    "train_idx": train_idx,
                    "val_idx": test_idx,
                    "similarity": round(sim, 3),
                    "train_q": splits['train'][train_idx].question[:100],
                    "val_q": splits['validation'][test_idx].question[:100]
                })
                report["total_flags"] += 1
    
    # Check train vs test
    for test_idx, test_q in test_qs:
        for train_idx, train_q in train_qs:
            sim = compute_jaccard_similarity(test_q, train_q)
            if sim >= similarity_threshold:
                report["train_test_duplicates"].append({
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "similarity": round(sim, 3),
                    "train_q": splits['train'][train_idx].question[:100],
                    "test_q": splits['test'][test_idx].question[:100]
                })
                report["total_flags"] += 1
    
    # Check validation vs test
    for test_idx, test_q in test_qs:
        for val_idx, val_q in val_qs:
            sim = compute_jaccard_similarity(test_q, val_q)
            if sim >= similarity_threshold:
                report["val_test_duplicates"].append({
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                    "similarity": round(sim, 3),
                    "val_q": splits['validation'][val_idx].question[:100],
                    "test_q": splits['test'][test_idx].question[:100]
                })
                report["total_flags"] += 1
    
    # Print summary
    print(f"   Train-Val near-duplicates: {len(report['train_val_duplicates'])}")
    print(f"   Train-Test near-duplicates: {len(report['train_test_duplicates'])}")
    print(f"   Val-Test near-duplicates: {len(report['val_test_duplicates'])}")
    print(f"   Total flags: {report['total_flags']}")
    
    if report['total_flags'] > 0:
        print(f"   ⚠️ WARNING: Found {report['total_flags']} potential near-duplicates!")
        print(f"   Review contamination_report.json for details.")
    else:
        print(f"   ✅ No near-duplicates detected. Clean split!")
    
    return report


def convert_to_sft(examples: list[DatasetExample]) -> list[TrainingSFTExample]:
    """Convert DatasetExamples to SFT training format."""
    sft_examples = []
    for ex in examples:
        sft = TrainingSFTExample(
            question=ex.question,
            context=ex.gold_passage or "",
            answer=ex.reference_answer,
            citation_ids=[ex.gold_passage_id] if ex.gold_passage_id else [],
            answerability=ex.answerability,
        )
        sft_examples.append(sft)
    return sft_examples


def convert_to_raft(
    examples: list[DatasetExample],
    all_passages: list[Passage],
    distractor_ratio: float = 0.5,
    num_distractors: int = 3,
    seed: int = 42,
) -> list[TrainingRAFTExample]:
    """Convert DatasetExamples to RAFT training format with distractors."""
    random.seed(seed)
    raft_examples = []

    for ex in examples:
        # Gold passage
        gold = Passage(
            passage_id=ex.gold_passage_id or f"{ex.question_id}_gold",
            doc_id=ex.doc_id or "unknown",
            text=ex.gold_passage or "",
        )

        # Sample distractors from other passages
        other_passages = [p for p in all_passages if p.passage_id != gold.passage_id]
        distractors = random.sample(other_passages, min(num_distractors, len(other_passages)))

        # Decide if this is a distractor-heavy example
        include_gold = random.random() > distractor_ratio

        if include_gold and ex.answerability != AnswerabilityLabel.UNANSWERABLE:
            relevant = [gold]
            answerability = ex.answerability
        else:
            relevant = []
            answerability = AnswerabilityLabel.UNANSWERABLE

        raft = TrainingRAFTExample(
            question=ex.question,
            relevant_passages=relevant,
            distractor_passages=distractors,
            answer=ex.reference_answer if answerability != AnswerabilityLabel.UNANSWERABLE
                   else "INSUFFICIENT_EVIDENCE: The provided passages do not contain sufficient information to answer this question.",
            supporting_passage_ids=[gold.passage_id] if include_gold else [],
            answerability=answerability,
        )
        raft_examples.append(raft)

    return raft_examples


def save_jsonl(examples: list, path: Path) -> None:
    """Save examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            if hasattr(ex, "model_dump"):
                f.write(json.dumps(ex.model_dump(), default=str) + "\n")
            else:
                f.write(json.dumps(ex, default=str) + "\n")
    print(f"   Saved {len(examples)} examples to {path}")


def validate_dataset(examples: list[DatasetExample]) -> dict:
    """Validate a dataset and return a report."""
    report = {
        "total": len(examples),
        "with_gold_passage": sum(1 for e in examples if e.gold_passage),
        "with_reference_answer": sum(1 for e in examples if e.reference_answer),
        "answerability_dist": {},
        "specialty_dist": {},
        "difficulty_dist": {},
        "duplicate_questions": 0,
    }

    # Distributions
    for ex in examples:
        ans = ex.answerability.value
        report["answerability_dist"][ans] = report["answerability_dist"].get(ans, 0) + 1
        if ex.specialty:
            report["specialty_dist"][ex.specialty] = report["specialty_dist"].get(ex.specialty, 0) + 1
        if ex.difficulty:
            diff = ex.difficulty.value if hasattr(ex.difficulty, "value") else str(ex.difficulty)
            report["difficulty_dist"][diff] = report["difficulty_dist"].get(diff, 0) + 1

    # Duplicate check
    questions = [ex.question.lower().strip() for ex in examples]
    report["duplicate_questions"] = len(questions) - len(set(questions))

    return report


def main():
    parser = argparse.ArgumentParser(description="Prepare MedFusion-RAFT datasets")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--include-pubmedqa", action="store_true", help="Include PubMedQA dataset")
    parser.add_argument("--build-raft", action="store_true", help="Build RAFT training data")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="TESTING ONLY: Use synthetic sample data instead of real datasets",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect examples
    all_examples = []

    # CRITICAL: Only use synthetic data in explicit smoke-test mode
    if args.smoke_test:
        print("\n⚠️  SMOKE TEST MODE: Using synthetic sample data")
        print("    This is for testing only - DO NOT use for real training/evaluation!\n")
        all_examples.extend(create_sample_medical_dataset())
    else:
        # Production mode: require at least one real dataset
        if not args.include_pubmedqa:
            print("\n❌ ERROR: No real datasets specified!")
            print("   Either:")
            print("   1. Add --include-pubmedqa for real PubMedQA data, OR")
            print("   2. Use --smoke-test flag for synthetic testing data\n")
            sys.exit(1)
        
        # Optionally include PubMedQA (only loaded in non-smoke-test mode)
        if args.include_pubmedqa:
            all_examples.extend(load_pubmedqa())

    if not all_examples:
        print("\n❌ ERROR: No examples collected!")
        sys.exit(1)

    print(f"\n📊 Total examples: {len(all_examples)}")

    # Validate
    report = validate_dataset(all_examples)
    print(f"\n📋 Dataset Validation Report:")
    print(f"   Total: {report['total']}")
    print(f"   With gold passage: {report['with_gold_passage']}")
    print(f"   Answerability: {report['answerability_dist']}")
    print(f"   Duplicates: {report['duplicate_questions']}")

    # Save validation report
    with open(output_dir / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Split
    print(f"\n✂️ Splitting dataset...")
    splits = split_by_document(
        all_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Detect contamination
    contamination_report = detect_near_duplicates(splits, similarity_threshold=0.85)
    contamination_path = output_dir / "contamination_report.json"
    with open(contamination_path, "w") as f:
        json.dump(contamination_report, f, indent=2)
    print(f"   Contamination report saved to {contamination_path}")

    # Save raw splits
    for split_name, split_examples in splits.items():
        save_jsonl(split_examples, output_dir / f"{split_name}.jsonl")

    # Convert to SFT format
    print(f"\n🔄 Converting to SFT format...")
    for split_name in ["train", "validation"]:
        sft_examples = convert_to_sft(splits[split_name])
        save_jsonl(sft_examples, output_dir / f"sft_{split_name}.jsonl")

    # Build RAFT data if requested
    if args.build_raft:
        print(f"\n🔄 Building RAFT training data...")
        # Collect all passages as potential distractors
        all_passages = []
        for ex in all_examples:
            if ex.gold_passage:
                all_passages.append(Passage(
                    passage_id=ex.gold_passage_id or f"{ex.question_id}_p0",
                    doc_id=ex.doc_id or "unknown",
                    text=ex.gold_passage,
                ))

        for split_name in ["train", "validation"]:
            raft_examples = convert_to_raft(
                splits[split_name],
                all_passages,
                seed=args.seed,
            )
            save_jsonl(raft_examples, output_dir / f"raft_{split_name}.jsonl")

            # Log distribution
            ans_dist = defaultdict(int)
            for ex in raft_examples:
                ans_dist[ex.answerability.value] += 1
            print(f"   RAFT {split_name} distribution: {dict(ans_dist)}")

    print(f"\n✅ Dataset preparation complete. Output: {output_dir}")
    
    if args.smoke_test:
        print("\n⚠️  REMINDER: Smoke test mode was used - synthetic data only!")
        print("    Do NOT use these files for real training or benchmarking.\n")


if __name__ == "__main__":
    main()

