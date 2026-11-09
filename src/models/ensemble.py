"""Multi-model ensemble for improved accuracy and calibration.

Implements ensemble strategies including:
- Majority voting
- Confidence-weighted averaging
- Selective prediction based on ensemble agreement
- Temperature scaling for calibration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.models.base import BaseGeneratorProvider, GenerationOutput, ParsedMedicalAnswer
from src.models.schemas import AnswerabilityLabel
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble."""
    aggregation_method: str = "confidence_weighted"  # "majority", "confidence_weighted", "selective"
    min_agreement_threshold: float = 0.6  # Minimum agreement for acceptance
    temperature_scaling: bool = True  # Apply temperature scaling to confidences
    temperature: float = 1.5  # Temperature parameter
    abstain_on_disagreement: bool = True  # Abstain if models disagree significantly


class MedicalEnsemble(BaseGeneratorProvider):
    """Ensemble of multiple medical QA models.
    
    Aggregates predictions from multiple generator models to improve
    accuracy, calibration, and uncertainty estimation.
    """
    
    def __init__(
        self,
        models: list[BaseGeneratorProvider],
        config: Optional[EnsembleConfig] = None,
    ):
        """Initialize ensemble.
        
        Args:
            models: List of generator models to ensemble
            config: Ensemble configuration
        """
        if not models:
            raise ValueError("Ensemble requires at least one model")
        
        # Use first model's config as base
        super().__init__(models[0].config)
        
        self.models = models
        self.config = config or EnsembleConfig()
        self._loaded = all(m.is_loaded for m in models)
        
        logger.info(
            f"Initialized ensemble with {len(models)} models, "
            f"aggregation={self.config.aggregation_method}"
        )
    
    def load(self) -> None:
        """Load all models in ensemble."""
        logger.info(f"Loading ensemble of {len(self.models)} models...")
        for i, model in enumerate(self.models):
            if not model.is_loaded:
                logger.info(f"Loading model {i+1}/{len(self.models)}")
                model.load()
        self._loaded = True
        logger.info("All ensemble models loaded")
    
    def unload(self) -> None:
        """Unload all models in ensemble."""
        logger.info("Unloading ensemble models...")
        for model in self.models:
            if model.is_loaded:
                model.unload()
        self._loaded = False
        logger.info("All ensemble models unloaded")
    
    def generate(self, prompt: str, **kwargs: Any) -> GenerationOutput:
        """Generate using ensemble aggregation.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Aggregated generation output
        """
        if not self.is_loaded:
            self.load()
        
        # Generate from all models
        outputs = []
        for i, model in enumerate(self.models):
            try:
                logger.debug(f"Generating with model {i+1}/{len(self.models)}")
                output = model.generate(prompt, **kwargs)
                outputs.append(output)
            except Exception as e:
                logger.warning(f"Model {i+1} generation failed: {e}")
                # Continue with other models
        
        if not outputs:
            raise RuntimeError("All ensemble models failed to generate")
        
        # Aggregate outputs
        aggregated = self._aggregate_outputs(outputs)
        
        logger.info(
            f"Ensemble generated {aggregated.output_tokens} tokens "
            f"in {aggregated.generation_time_ms:.1f}ms"
        )
        
        return aggregated
    
    def generate_answer(
        self,
        question: str,
        context: str | list[str],
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate medical answer using ensemble.
        
        Args:
            question: Medical question
            context: Evidence context
            **kwargs: Generation parameters
            
        Returns:
            Aggregated answer
        """
        if not self.is_loaded:
            self.load()
        
        # Generate from all models
        outputs = []
        parsed_answers = []
        
        for i, model in enumerate(self.models):
            try:
                logger.debug(f"Generating answer with model {i+1}/{len(self.models)}")
                output = model.generate_answer(question, context, **kwargs)
                outputs.append(output)
                
                # Parse the answer
                if hasattr(model, 'parse_medical_answer'):
                    parsed = model.parse_medical_answer(output.text)
                    parsed_answers.append(parsed)
                
            except Exception as e:
                logger.warning(f"Model {i+1} answer generation failed: {e}")
        
        if not outputs:
            raise RuntimeError("All ensemble models failed to generate answers")
        
        # Aggregate answers
        aggregated = self._aggregate_medical_answers(outputs, parsed_answers)
        
        return aggregated
    
    def _aggregate_outputs(self, outputs: list[GenerationOutput]) -> GenerationOutput:
        """Aggregate multiple generation outputs.
        
        Args:
            outputs: List of outputs from ensemble models
            
        Returns:
            Aggregated output
        """
        if self.config.aggregation_method == "majority":
            return self._majority_voting(outputs)
        elif self.config.aggregation_method == "confidence_weighted":
            return self._confidence_weighted(outputs)
        elif self.config.aggregation_method == "selective":
            return self._selective_prediction(outputs)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
    
    def _majority_voting(self, outputs: list[GenerationOutput]) -> GenerationOutput:
        """Aggregate using majority voting.
        
        Args:
            outputs: List of outputs
            
        Returns:
            Majority output
        """
        # Simple: take most common output
        # For medical answers, this is more complex - need semantic similarity
        
        # For now, return first output with metadata about voting
        result = outputs[0]
        result.metadata["ensemble_method"] = "majority_voting"
        result.metadata["ensemble_size"] = len(outputs)
        result.metadata["agreement_score"] = 1.0 / len(outputs)  # Placeholder
        
        return result
    
    def _confidence_weighted(self, outputs: list[GenerationOutput]) -> GenerationOutput:
        """Aggregate using confidence-weighted averaging.
        
        Args:
            outputs: List of outputs
            
        Returns:
            Confidence-weighted output
        """
        # Extract confidences (if available in metadata)
        confidences = []
        for output in outputs:
            conf = output.metadata.get("confidence", 1.0)
            if self.config.temperature_scaling:
                # Apply temperature scaling
                conf = conf ** (1.0 / self.config.temperature)
            confidences.append(conf)
        
        # Normalize confidences
        total_conf = sum(confidences)
        if total_conf > 0:
            weights = [c / total_conf for c in confidences]
        else:
            weights = [1.0 / len(outputs)] * len(outputs)
        
        # Select output with highest confidence
        max_idx = np.argmax(confidences)
        result = outputs[max_idx]
        
        # Add ensemble metadata
        result.metadata["ensemble_method"] = "confidence_weighted"
        result.metadata["ensemble_size"] = len(outputs)
        result.metadata["ensemble_weights"] = weights
        result.metadata["ensemble_confidence_mean"] = float(np.mean(confidences))
        result.metadata["ensemble_confidence_std"] = float(np.std(confidences))
        
        logger.debug(
            f"Confidence-weighted: selected model with weight {weights[max_idx]:.3f}"
        )
        
        return result
    
    def _selective_prediction(self, outputs: list[GenerationOutput]) -> GenerationOutput:
        """Aggregate with selective prediction (abstain on disagreement).
        
        Args:
            outputs: List of outputs
            
        Returns:
            Aggregated output or abstention signal
        """
        # Compute agreement between outputs
        # (Simplified - would use semantic similarity in practice)
        
        texts = [o.text for o in outputs]
        
        # Simple word overlap as agreement measure
        all_words = [set(t.lower().split()) for t in texts]
        
        if len(all_words) < 2:
            agreement = 1.0
        else:
            # Average pairwise Jaccard similarity
            agreements = []
            for i in range(len(all_words)):
                for j in range(i + 1, len(all_words)):
                    intersection = len(all_words[i] & all_words[j])
                    union = len(all_words[i] | all_words[j])
                    if union > 0:
                        agreements.append(intersection / union)
            agreement = np.mean(agreements) if agreements else 0.0
        
        result = outputs[0]
        result.metadata["ensemble_method"] = "selective"
        result.metadata["ensemble_size"] = len(outputs)
        result.metadata["ensemble_agreement"] = float(agreement)
        
        # Check if agreement meets threshold
        if self.config.abstain_on_disagreement and agreement < self.config.min_agreement_threshold:
            logger.warning(
                f"Low ensemble agreement ({agreement:.3f} < {self.config.min_agreement_threshold}), "
                "considering abstention"
            )
            result.metadata["ensemble_abstain_recommended"] = True
        
        return result
    
    def _aggregate_medical_answers(
        self,
        outputs: list[GenerationOutput],
        parsed_answers: list[ParsedMedicalAnswer],
    ) -> GenerationOutput:
        """Aggregate medical answers with domain-specific logic.
        
        Args:
            outputs: Raw generation outputs
            parsed_answers: Parsed medical answers
            
        Returns:
            Aggregated medical answer
        """
        if not parsed_answers:
            return self._aggregate_outputs(outputs)
        
        # Extract answerability consensus
        answerabilities = [a.answerability for a in parsed_answers]
        answerability_counts = {}
        for ans in answerabilities:
            answerability_counts[ans] = answerability_counts.get(ans, 0) + 1
        
        # Majority answerability
        majority_answerability = max(answerability_counts, key=answerability_counts.get)
        
        # Extract citations consensus
        all_citations = []
        for parsed in parsed_answers:
            all_citations.extend(parsed.citations)
        
        # Count citation frequency
        citation_counts = {}
        for cit in all_citations:
            citation_counts[cit] = citation_counts.get(cit, 0) + 1
        
        # Keep citations mentioned by majority of models
        min_citation_count = len(parsed_answers) // 2 + 1
        consensus_citations = [
            cit for cit, count in citation_counts.items()
            if count >= min_citation_count
        ]
        
        # Extract confidence values
        confidences = [a.confidence for a in parsed_answers]
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Select answer with highest confidence
        max_conf_idx = np.argmax(confidences)
        result = outputs[max_conf_idx]
        
        # Enhance metadata
        result.metadata["ensemble_method"] = self.config.aggregation_method
        result.metadata["ensemble_size"] = len(outputs)
        result.metadata["ensemble_answerability"] = majority_answerability.value
        result.metadata["ensemble_citations"] = consensus_citations
        result.metadata["ensemble_confidence_mean"] = float(mean_confidence)
        result.metadata["ensemble_confidence_std"] = float(std_confidence)
        result.metadata["ensemble_agreement_answerability"] = answerability_counts[majority_answerability] / len(answerabilities)
        
        logger.info(
            f"Ensemble medical answer: answerability={majority_answerability.value}, "
            f"confidence={mean_confidence:.3f}±{std_confidence:.3f}, "
            f"citations={len(consensus_citations)}"
        )
        
        return result
    
    def get_memory_usage(self) -> dict[str, float]:
        """Get total memory usage across ensemble.
        
        Returns:
            Dictionary with memory statistics
        """
        total_stats = {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "system_ram_mb": 0.0,
        }
        
        for model in self.models:
            stats = model.get_memory_usage()
            for key in total_stats:
                total_stats[key] += stats.get(key, 0.0)
        
        total_stats["num_models"] = len(self.models)
        
        return total_stats
