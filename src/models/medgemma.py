"""MedGemma generator provider implementation for medical question answering."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional, Union

from src.models.base import (
    AnswerabilityStatus,
    BaseGeneratorProvider,
    GenerationOutput,
    ModelConfig,
    ModelInferenceError,
    ModelLoadError,
    ParsedMedicalAnswer,
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

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None  # type: ignore[assignment]
    PEFT_AVAILABLE = False


DEFAULT_SYSTEM_PROMPT = (
    "You are a medical question answering assistant. Ground your answers in the provided evidence. "
    "Cite passage IDs. If evidence is insufficient, say so."
)

DEFAULT_GENERATION_PROMPT_TEMPLATE = """Given the following medical question and evidence passages, provide a structured answer.

Question: {question}

Evidence:
{formatted_passages}

Instructions:
1. Answer the question based ONLY on the provided evidence
2. Cite specific passage IDs [P1], [P2], etc. for each claim
3. If the evidence is insufficient, state "INSUFFICIENT_EVIDENCE"
4. If evidence conflicts, note the contradiction
5. Be concise and medically precise

Respond in this format:
ANSWER: <your answer>
CITATIONS: <list of passage IDs used>
ANSWERABILITY: <answerable|partially_answerable|unanswerable>
CONFIDENCE: <0.0-1.0>"""


class MedGemmaProvider(BaseGeneratorProvider):
    """Model provider for MedGemma / Gemma causal language models.

    Supports 4-bit/8-bit QLoRA quantization via BitsAndBytes and PEFT adapter weights.
    """

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Initialize MedGemma provider."""
        super().__init__(config)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.prompt_template = prompt_template or DEFAULT_GENERATION_PROMPT_TEMPLATE

    def _build_bnb_config(self) -> Optional[Any]:
        """Construct BitsAndBytesConfig based on ModelConfig."""
        if not self.config.quantization_bits:
            return None
        if not TRANSFORMERS_AVAILABLE or BitsAndBytesConfig is None:
            raise ModelLoadError("transformers with bitsandbytes is required for quantization.")

        compute_dtype = getattr(torch, self.config.compute_dtype, torch.bfloat16)
        if self.config.quantization_bits == 4:
            logger.info("Configuring 4-bit quantization (type=%s)", self.config.quant_type)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_double_quant,
            )
        elif self.config.quantization_bits == 8:
            logger.info("Configuring 8-bit quantization")
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"Unsupported quantization_bits: {self.config.quantization_bits}")

    def load(self) -> None:
        """Load tokenizer and MedGemma model into memory."""
        if self._loaded:
            logger.debug("MedGemma provider is already loaded.")
            return
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ModelLoadError("PyTorch and Transformers are required to load MedGemmaProvider.")

        logger.info("Loading MedGemma: %s (device_map=%s)", self.config.model_id, self.config.device_map)
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

            if self.config.adapter_path:
                if not PEFT_AVAILABLE or PeftModel is None:
                    raise ModelLoadError("peft is required to load adapter weights.")
                logger.info("Loading PEFT adapter from %s", self.config.adapter_path)
                self._model = PeftModel.from_pretrained(self._model, self.config.adapter_path)

            self._model.eval()
            self._loaded = True
            logger.info("MedGemma loaded. Memory: %s", self.get_memory_usage())
        except Exception as exc:
            self.unload()
            logger.error("Failed to load MedGemma: %s", exc, exc_info=True)
            raise ModelLoadError(f"Could not load MedGemma {self.config.model_id}: {exc}") from exc

    def unload(self) -> None:
        """Unload model and tokenizer, freeing GPU memory."""
        logger.info("Unloading MedGemma model...")
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("MedGemma unloaded successfully.")

    def _apply_chat_template(self, prompt: str) -> str:
        """Format prompt with Gemma turn tokens or tokenizer chat template."""
        messages = [{"role": "user", "content": f"{self.system_prompt}\n\n{prompt}"}]
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"<start_of_turn>user\n{self.system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def generate(self, prompt: str, **kwargs: Any) -> GenerationOutput:
        """Generate text completion from prompt."""
        if not self.is_loaded:
            self.load()

        formatted_prompt = self._apply_chat_template(prompt)
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

            logger.info("MedGemma: %d in / %d out tokens in %.2f ms", input_tokens, output_tokens, elapsed_ms)
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
            logger.error("MedGemma generation error: %s", exc, exc_info=True)
            raise ModelInferenceError(f"MedGemma generation error: {exc}") from exc

    def _format_evidence_passages(self, context: Union[str, list[str]]) -> str:
        """Format evidence passages with passage ID tags."""
        if isinstance(context, str):
            return context if "[P" in context else f"[P1] {context}"
        return "\n\n".join(
            p.strip() if p.strip().startswith("[P") else f"[P{i+1}] {p.strip()}"
            for i, p in enumerate(context)
        )

    def generate_answer(
        self,
        question: str,
        context: Union[str, list[str]],
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate a structured medical answer based on question and evidence."""
        formatted_passages = self._format_evidence_passages(context)
        prompt = self.prompt_template.format(question=question, formatted_passages=formatted_passages)
        raw_output = self.generate(prompt, **kwargs)
        parsed = self.parse_medical_answer(raw_output.text)
        raw_output.metadata["parsed_answer"] = parsed.model_dump()
        return raw_output

    @staticmethod
    def parse_medical_answer(text: str) -> ParsedMedicalAnswer:
        """Parse structured fields (ANSWER, CITATIONS, ANSWERABILITY, CONFIDENCE)."""
        answer = text
        citations: list[str] = []
        answerability = AnswerabilityStatus.ANSWERABLE
        confidence = 1.0

        ans_m = re.search(r"ANSWER:\s*(.*?)(?=\nCITATIONS:|\nANSWERABILITY:|\nCONFIDENCE:|$)", text, re.DOTALL | re.I)
        if ans_m:
            answer = ans_m.group(1).strip()

        cit_m = re.search(r"CITATIONS:\s*([^\n]*)", text, re.I)
        if cit_m:
            found_ids = re.findall(r"\[P\d+\]|P\d+", cit_m.group(1).strip())
            citations = [fid if fid.startswith("[") else f"[{fid}]" for fid in found_ids]

        ansb_m = re.search(r"ANSWERABILITY:\s*([a-zA-Z_]+)", text, re.I)
        if ansb_m:
            val = ansb_m.group(1).strip().lower()
            if "unanswerable" in val:
                answerability = AnswerabilityStatus.UNANSWERABLE
            elif "partially" in val:
                answerability = AnswerabilityStatus.PARTIALLY_ANSWERABLE

        conf_m = re.search(r"CONFIDENCE:\s*([\d\.]+)", text, re.I)
        if conf_m:
            try:
                confidence = max(0.0, min(1.0, float(conf_m.group(1).strip())))
            except ValueError:
                confidence = 1.0

        return ParsedMedicalAnswer(
            answer=answer,
            citations=citations,
            answerability=answerability,
            confidence=confidence,
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
