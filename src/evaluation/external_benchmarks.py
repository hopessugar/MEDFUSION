"""External medical benchmark integration for MedFusion-RAFT.

Supports standard medical QA benchmarks including MedQA, MedMCQA, MMLU Medical,
LiveQA, and PubMedQA for comparison with published baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.models.schemas import DatasetExample, QuestionDifficulty, AnswerabilityLabel
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional HuggingFace datasets import
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets not available. Install with: pip install datasets")


@dataclass
class BenchmarkMetadata:
    """Metadata for a medical QA benchmark."""
    name: str
    description: str
    num_examples: int
    citation: str
    license: str
    url: str
    baseline_accuracy: Optional[float] = None
    human_performance: Optional[float] = None


class MedQALoader:
    """Loader for MedQA (USMLE-style questions)."""
    
    METADATA = BenchmarkMetadata(
        name="MedQA",
        description="USMLE-style multiple choice medical questions",
        num_examples=12723,
        citation="Jin et al. 2021. What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams",
        license="MIT",
        url="https://huggingface.co/datasets/bigbio/med_qa",
        baseline_accuracy=0.36,  # GPT-3.5
        human_performance=0.87,  # Medical professionals
    )
    
    def load(self, split: str = "test", max_examples: Optional[int] = None) -> list[DatasetExample]:
        """Load MedQA dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_examples: Optional limit on number of examples
            
        Returns:
            List of DatasetExample objects
        """
        if not DATASETS_AVAILABLE:
            logger.error("datasets library required for MedQA")
            return []
        
        logger.info(f"Loading MedQA {split} split...")
        try:
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa", split=split, trust_remote_code=True)
            
            examples = []
            for i, item in enumerate(dataset):
                if max_examples and i >= max_examples:
                    break
                
                question_id = item.get("id", f"medqa_{i}")
                question_text = item.get("question", "")
                
                # MedQA has options and correct answer
                choices = item.get("choices", [])
                answer_idx = item.get("answer_idx", 0)
                
                if choices and 0 <= answer_idx < len(choices):
                    reference_answer = choices[answer_idx]
                else:
                    reference_answer = item.get("answer", "")
                
                examples.append(DatasetExample(
                    question_id=f"medqa_{question_id}",
                    question=question_text,
                    reference_answer=reference_answer,
                    specialty="general_medicine",
                    difficulty=QuestionDifficulty.HARD,
                    answerability=AnswerabilityLabel.ANSWERABLE,
                    source_url="https://huggingface.co/datasets/bigbio/med_qa",
                    review_status="benchmark",
                ))
            
            logger.info(f"Loaded {len(examples)} MedQA examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load MedQA: {e}")
            return []


class MedMCQALoader:
    """Loader for MedMCQA (Indian medical entrance exams)."""
    
    METADATA = BenchmarkMetadata(
        name="MedMCQA",
        description="Indian medical entrance exam questions covering 2400+ medical topics",
        num_examples=194000,
        citation="Pal et al. 2022. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering",
        license="CC BY-NC-SA 4.0",
        url="https://huggingface.co/datasets/medmcqa",
        baseline_accuracy=0.40,
        human_performance=0.90,
    )
    
    def load(self, split: str = "validation", max_examples: Optional[int] = None) -> list[DatasetExample]:
        """Load MedMCQA dataset."""
        if not DATASETS_AVAILABLE:
            logger.error("datasets library required for MedMCQA")
            return []
        
        logger.info(f"Loading MedMCQA {split} split...")
        try:
            dataset = load_dataset("medmcqa", split=split, trust_remote_code=True)
            
            examples = []
            specialty_map = {
                "Anatomy": "anatomy",
                "Physiology": "physiology",
                "Biochemistry": "biochemistry",
                "Pharmacology": "pharmacology",
                "Medicine": "internal_medicine",
                "Surgery": "surgery",
                "Pediatrics": "pediatrics",
                "Gynaecology & Obstetrics": "obgyn",
            }
            
            for i, item in enumerate(dataset):
                if max_examples and i >= max_examples:
                    break
                
                question_id = item.get("id", f"medmcqa_{i}")
                question_text = item.get("question", "")
                
                # Get correct answer
                opa = item.get("opa", "")
                opb = item.get("opb", "")
                opc = item.get("opc", "")
                opd = item.get("opd", "")
                cop = item.get("cop", 0)  # Correct option (0-3)
                
                options = [opa, opb, opc, opd]
                reference_answer = options[cop] if 0 <= cop < len(options) else ""
                
                subject = item.get("subject_name", "general_medicine")
                specialty = specialty_map.get(subject, "general_medicine")
                
                examples.append(DatasetExample(
                    question_id=f"medmcqa_{question_id}",
                    question=question_text,
                    reference_answer=reference_answer,
                    specialty=specialty,
                    difficulty=QuestionDifficulty.MEDIUM,
                    answerability=AnswerabilityLabel.ANSWERABLE,
                    source_url="https://huggingface.co/datasets/medmcqa",
                    review_status="benchmark",
                ))
            
            logger.info(f"Loaded {len(examples)} MedMCQA examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load MedMCQA: {e}")
            return []


class MMLUMedicalLoader:
    """Loader for MMLU Medical subset."""
    
    METADATA = BenchmarkMetadata(
        name="MMLU Medical",
        description="Medical subset of Massive Multitask Language Understanding benchmark",
        num_examples=1089,
        citation="Hendrycks et al. 2021. Measuring Massive Multitask Language Understanding",
        license="MIT",
        url="https://huggingface.co/datasets/cais/mmlu",
        baseline_accuracy=0.52,
        human_performance=0.89,
    )
    
    MEDICAL_SUBJECTS = [
        "anatomy",
        "clinical_knowledge",
        "college_biology",
        "college_medicine",
        "medical_genetics",
        "professional_medicine",
    ]
    
    def load(self, split: str = "test", max_examples: Optional[int] = None) -> list[DatasetExample]:
        """Load MMLU Medical subset."""
        if not DATASETS_AVAILABLE:
            logger.error("datasets library required for MMLU")
            return []
        
        logger.info(f"Loading MMLU Medical {split} split...")
        examples = []
        
        try:
            for subject in self.MEDICAL_SUBJECTS:
                logger.info(f"Loading MMLU subject: {subject}")
                dataset = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
                
                for i, item in enumerate(dataset):
                    if max_examples and len(examples) >= max_examples:
                        break
                    
                    question = item.get("question", "")
                    choices = item.get("choices", [])
                    answer_idx = item.get("answer", 0)
                    
                    if choices and 0 <= answer_idx < len(choices):
                        reference_answer = choices[answer_idx]
                    else:
                        reference_answer = ""
                    
                    examples.append(DatasetExample(
                        question_id=f"mmlu_{subject}_{i}",
                        question=question,
                        reference_answer=reference_answer,
                        specialty=subject,
                        difficulty=QuestionDifficulty.HARD,
                        answerability=AnswerabilityLabel.ANSWERABLE,
                        source_url=f"https://huggingface.co/datasets/cais/mmlu",
                        review_status="benchmark",
                    ))
            
            logger.info(f"Loaded {len(examples)} MMLU Medical examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load MMLU Medical: {e}")
            return []


class PubMedQALoader:
    """Loader for full PubMedQA dataset."""
    
    METADATA = BenchmarkMetadata(
        name="PubMedQA",
        description="Biomedical question answering from PubMed abstracts",
        num_examples=1000,
        citation="Jin et al. 2019. PubMedQA: A Dataset for Biomedical Research Question Answering",
        license="MIT",
        url="https://huggingface.co/datasets/qiaojin/PubMedQA",
        baseline_accuracy=0.55,
        human_performance=0.78,
    )
    
    def load(self, split: str = "pqa_labeled", max_examples: Optional[int] = None) -> list[DatasetExample]:
        """Load PubMedQA dataset."""
        if not DATASETS_AVAILABLE:
            logger.error("datasets library required for PubMedQA")
            return []
        
        logger.info(f"Loading PubMedQA {split} split...")
        try:
            # PubMedQA requires config name
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)
            
            examples = []
            for i, item in enumerate(dataset):
                if max_examples and i >= max_examples:
                    break
                
                question = item.get("question", "")
                context_dict = item.get("context", {})
                contexts = context_dict.get("contexts", [])
                answer = item.get("long_answer", "")
                final_decision = item.get("final_decision", "yes")
                
                # Determine answerability
                if final_decision == "maybe":
                    answerability = AnswerabilityLabel.PARTIALLY_ANSWERABLE
                else:
                    answerability = AnswerabilityLabel.ANSWERABLE
                
                # Combine contexts as evidence
                evidence = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
                
                examples.append(DatasetExample(
                    question_id=f"pubmedqa_{i}",
                    question=question,
                    reference_answer=answer,
                    gold_passage=evidence[:1000] if evidence else None,
                    specialty="biomedical_research",
                    difficulty=QuestionDifficulty.MEDIUM,
                    answerability=answerability,
                    source_url="https://huggingface.co/datasets/qiaojin/PubMedQA",
                    review_status="benchmark",
                ))
            
            logger.info(f"Loaded {len(examples)} PubMedQA examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load PubMedQA: {e}")
            return []


class BenchmarkRegistry:
    """Central registry for all medical QA benchmarks."""
    
    def __init__(self):
        self.benchmarks = {
            "medqa": MedQALoader(),
            "medmcqa": MedMCQALoader(),
            "mmlu_medical": MMLUMedicalLoader(),
            "pubmedqa": PubMedQALoader(),
        }
    
    def list_benchmarks(self) -> dict[str, BenchmarkMetadata]:
        """List all available benchmarks with metadata."""
        return {
            name: loader.METADATA 
            for name, loader in self.benchmarks.items()
        }
    
    def load_benchmark(
        self, 
        benchmark_name: str, 
        split: str = "test",
        max_examples: Optional[int] = None
    ) -> list[DatasetExample]:
        """Load a specific benchmark.
        
        Args:
            benchmark_name: Name of benchmark ('medqa', 'medmcqa', 'mmlu_medical', 'pubmedqa')
            split: Dataset split
            max_examples: Optional limit on examples
            
        Returns:
            List of DatasetExample objects
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name}. "
                f"Available: {list(self.benchmarks.keys())}"
            )
        
        loader = self.benchmarks[benchmark_name]
        return loader.load(split=split, max_examples=max_examples)
    
    def load_all_benchmarks(
        self, 
        split: str = "test",
        max_examples_per_benchmark: Optional[int] = None
    ) -> dict[str, list[DatasetExample]]:
        """Load all benchmarks.
        
        Returns:
            Dictionary mapping benchmark names to examples
        """
        results = {}
        for name in self.benchmarks:
            logger.info(f"Loading benchmark: {name}")
            results[name] = self.load_benchmark(
                name, 
                split=split, 
                max_examples=max_examples_per_benchmark
            )
        return results
    
    def save_benchmark_to_jsonl(
        self, 
        benchmark_name: str,
        output_path: Path,
        split: str = "test",
        max_examples: Optional[int] = None
    ) -> None:
        """Load and save benchmark to JSONL format."""
        examples = self.load_benchmark(benchmark_name, split, max_examples)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.model_dump()) + "\n")
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
