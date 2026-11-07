"""Qwen causal LM verification provider for medical factual consistency check."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional, Union

from src.models.base import (
    BaseVerifierProvider,
    ClaimVerificationStatus,
    GenerationOutput,
    ModelConfig,
    ModelInferenceError,
    ModelLoadError,
    OverallAssessmentStatus,
    ParsedVerificationResult,
    VerificationClaim,
)

logger = logging.getLogger(__name__)

# Optional dependencies imported gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


DEFAULT_VERIFIER_SYSTEM_PROMPT = (
    "You are an independent medical answer verifier. Your task is to verify whether "
    "a generated medical answer is supported by the provided evidence."
)

DEFAULT_VERIFICATION_PROMPT_TEMPLATE = """You are an independent medical answer verifier. Your task is to verify whether a generated medical answer is supported by the provided evidence.

Question: {question}

Generated Answer: {answer}

Evidence Passages:
{passages}

Instructions:
1. Break the answer into atomic factual claims
2. For each claim, determine if it is SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or CONTRADICTED by the evidence
3. Check that cited passage IDs actually contain the claimed information
4. Identify any claims not supported by any evidence passage

Respond in this exact format:
CLAIMS:
- Claim: <claim text> | Status: <status> | Evidence: <passage_id or NONE> | Reasoning: <brief explanation>

OVERALL_ASSESSMENT: <VERIFIED|PARTIALLY_VERIFIED|NOT_VERIFIED>
SUPPORT_SCORE: <0.0-1.0>
ISSUES: <list any problems found>"""


class QwenVerifierProvider(BaseVerifierProvider):
    """Qwen causal language model verifier provider for factual consistency checking."""

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Initialize Qwen verifier provider."""
        super().__init__(config)
        self.system_prompt = system_prompt or DEFAULT_VERIFIER_SYSTEM_PROMPT
        self.prompt_template = prompt_template or DEFAULT_VERIFICATION_PROMPT_TEMPLATE

    def _build_bnb_config(self) -> Optional[Any]:
        """Construct BitsAndBytesConfig based on ModelConfig."""
        if not self.config.quantization_bits:
            return None
        if not TRANSFORMERS_AVAILABLE or BitsAndBytesConfig is None:
            raise ModelLoadError("transformers with bitsandbytes is required for quantization.")

        compute_dtype = getattr(torch, self.config.compute_dtype, torch.bfloat16)
        if self.config.quantization_bits == 4:
            logger.info("Configuring Qwen 4-bit quant (type=%s)", self.config.quant_type)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_double_quant,
            )
        elif self.config.quantization_bits == 8:
            logger.info("Configuring Qwen 8-bit quant")
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"Unsupported quantization_bits: {self.config.quantization_bits}")

    def load(self) -> None:
        """Load tokenizer and Qwen model into memory."""
        if self._loaded:
            logger.debug("Qwen verifier is already loaded.")
            return
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ModelLoadError("PyTorch and Transformers are required to load QwenVerifierProvider.")

        logger.info("Loading Qwen verifier: %s (device_map=%s)", self.config.model_id, self.config.device_map)
        try:
            bnb_config = self._build_bnb_config()
            load_kwargs: dict[str, Any] = {
                "revision": self.config.revision,
                "trust_remote_code": self.config.trust_remote_code,
                "device_map": self.config.device_map,
            }
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            elif self.config.compute_dtype and hasattr(torch, self.config.compute_dtype):
                load_kwargs["torch_dtype"] = getattr(torch, self.config.compute_dtype)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)
            self._model.eval()
            self._loaded = True
            logger.info("Qwen verifier loaded. Memory: %s", self.get_memory_usage())
        except Exception as exc:
            self.unload()
            logger.error("Failed to load Qwen verifier: %s", exc, exc_info=True)
            raise ModelLoadError(f"Could not load Qwen verifier {self.config.model_id}: {exc}") from exc

    def unload(self) -> None:
        """Unload model and tokenizer, releasing GPU resources."""
        logger.info("Unloading Qwen verifier model...")
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Qwen verifier unloaded successfully.")

    def _apply_chatml_template(self, prompt: str) -> str:
        """Format prompt using ChatML template for Qwen."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def generate(self, prompt: str, **kwargs: Any) -> GenerationOutput:
        """Generate text completion from prompt."""
        if not self.is_loaded:
            self.load()

        formatted_prompt = self._apply_chatml_template(prompt)
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        do_sample = kwargs.get("do_sample", self.config.do_sample)
        repetition_penalty = kwargs.get("repetition_penalty", self.config.repetition_penalty)

        start_time = time.perf_counter()
        try:
            inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
            if hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            input_tokens = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0][input_tokens:]
            output_tokens = len(gen_tokens)
            output_text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info("Qwen: %d in / %d out tokens in %.2f ms", input_tokens, output_tokens, elapsed_ms)
            return GenerationOutput(
                text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time_ms=elapsed_ms,
                metadata={
                    "model_id": self.config.model_id,
                    "tokens_per_sec": (output_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
                },
            )
        except Exception as exc:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error("Qwen verification generation failed: %s", exc, exc_info=True)
            raise ModelInferenceError(f"Qwen verification generation failed: {exc}") from exc

    def _format_passages(self, evidence: Union[str, list[str]]) -> str:
        """Format evidence list into passage blocks."""
        if isinstance(evidence, str):
            return evidence
        return "\n\n".join(
            p.strip() if p.strip().startswith("[P") else f"[P{i+1}] {p.strip()}"
            for i, p in enumerate(evidence)
        )

    def verify_answer(
        self,
        question: str,
        answer: str,
        evidence: Union[str, list[str]],
        **kwargs: Any,
    ) -> GenerationOutput:
        """Verify an answer against evidence passages."""
        passages_text = self._format_passages(evidence)
        prompt = self.prompt_template.format(question=question, answer=answer, passages=passages_text)
        raw_output = self.generate(prompt, **kwargs)
        parsed = self.parse_verification_output(raw_output.text)
        raw_output.metadata["parsed_verification"] = parsed.model_dump()
        return raw_output

    @staticmethod
    def parse_verification_output(text: str) -> ParsedVerificationResult:
        """Parse structured verification output from model response."""
        claims: list[VerificationClaim] = []
        overall = OverallAssessmentStatus.VERIFIED
        support_score = 1.0
        issues: list[str] = []

        claim_lines = re.findall(r"-\s*Claim:\s*(.*?)(?=\n-|\n[A-Z_]+:|$)", text, re.DOTALL)
        for raw_claim_block in claim_lines:
            parts = re.split(r"\s*\|\s*", raw_claim_block.strip())
            claim_text = ""
            status = ClaimVerificationStatus.SUPPORTED
            evidence = "NONE"
            reasoning = ""

            for part in parts:
                p_lower = part.lower()
                if p_lower.startswith("claim:"):
                    claim_text = part[6:].strip()
                elif p_lower.startswith("status:"):
                    st = part[7:].strip().upper()
                    if "PARTIAL" in st:
                        status = ClaimVerificationStatus.PARTIALLY_SUPPORTED
                    elif "UNSUPPORTED" in st:
                        status = ClaimVerificationStatus.UNSUPPORTED
                    elif "CONTRADICT" in st:
                        status = ClaimVerificationStatus.CONTRADICTED
                    else:
                        status = ClaimVerificationStatus.SUPPORTED
                elif p_lower.startswith("evidence:"):
                    evidence = part[9:].strip()
                elif p_lower.startswith("reasoning:"):
                    reasoning = part[10:].strip()
                elif not claim_text:
                    claim_text = part.strip()

            if claim_text:
                claims.append(VerificationClaim(claim=claim_text, status=status, evidence=evidence, reasoning=reasoning))

        ov_m = re.search(r"OVERALL_ASSESSMENT:\s*([A-Za-z_]+)", text, re.I)
        if ov_m:
            ov = ov_m.group(1).strip().upper()
            if "PARTIAL" in ov:
                overall = OverallAssessmentStatus.PARTIALLY_VERIFIED
            elif "NOT" in ov or "UNVERIFIED" in ov:
                overall = OverallAssessmentStatus.NOT_VERIFIED

        score_m = re.search(r"SUPPORT_SCORE:\s*([\d\.]+)", text, re.I)
        if score_m:
            try:
                support_score = max(0.0, min(1.0, float(score_m.group(1).strip())))
            except ValueError:
                support_score = 1.0
        elif claims:
            sup_count = sum(1 for c in claims if c.status == ClaimVerificationStatus.SUPPORTED) + 0.5 * sum(
                1 for c in claims if c.status == ClaimVerificationStatus.PARTIALLY_SUPPORTED
            )
            support_score = round(sup_count / len(claims), 2)

        issues_m = re.search(r"ISSUES:\s*(.*?)(?=\n[A-Z_]+:|$)", text, re.DOTALL | re.I)
        if issues_m:
            raw = issues_m.group(1).strip()
            if raw.lower() not in {"none", "none.", "n/a", ""}:
                issues = [item.strip() for item in re.split(r"[\n;]|\s*-\s*", raw) if item.strip()]

        return ParsedVerificationResult(
            claims=claims,
            overall_assessment=overall,
            support_score=support_score,
            issues=issues,
            raw_text=text,
        )

    def get_memory_usage(self) -> dict[str, float]:
        """Return GPU and process memory stats in MB."""
        stats = {"gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0, "system_ram_mb": 0.0}
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            stats["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        try:
            import psutil
            stats["system_ram_mb"] = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass
        return stats
