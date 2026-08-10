from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from collections.abc import Callable
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_AZURE_ENDPOINT = "https://csnf-singularity-aoai-eastus2.openai.azure.com/"
DEFAULT_AZURE_DEPLOYMENT = "gpt-5.6-sol"
DEFAULT_MANAGED_IDENTITY_CLIENT_ID = "e6162a0d-e540-4454-995f-30bcb97f35b4"


@dataclass(frozen=True)
class VLMScoreResult:
    reward: float
    label: float
    status: str
    output_text: str | None
    response_id: str | None
    attempt_count: int
    latency_ms: float
    error_type: str | None = None


def load_env_files() -> None:
    project_root = Path(__file__).resolve().parents[2]
    repository_root = Path(__file__).resolve().parents[4]
    for env_path in (repository_root / ".env", project_root / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#") or "=" not in value:
                continue
            key, raw_value = value.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip().strip('"').strip("'")
            if key and raw_value:
                os.environ.setdefault(key, raw_value)


def _get_bearer_token_provider(credential: Any, *scopes: str) -> Callable[[], str]:
    from azure.core.pipeline import PipelineContext, PipelineRequest
    from azure.core.pipeline.policies import BearerTokenCredentialPolicy
    from azure.core.rest import HttpRequest

    policy = BearerTokenCredentialPolicy(credential, *scopes)

    def wrapper() -> str:
        request = PipelineRequest(HttpRequest("GET", "https://fakeurl"), PipelineContext(None))
        policy.on_request(request)
        return request.http_request.headers["Authorization"][len("Bearer ") :]

    return wrapper


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def _parse_label_with_status(text: str, fallback_label: float = 2.5) -> tuple[float, bool]:
    try:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            payload = json.loads(match.group(0))
            label = payload.get("evaluation", {}).get("label")
            if label is not None:
                return max(0.0, min(5.0, float(label))), True
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        pass

    fallback = re.search(r'"label"\s*:\s*"?([0-5])"?', text)
    if fallback:
        return float(fallback.group(1)), True
    return max(0.0, min(5.0, fallback_label)), False


def parse_label(text: str, fallback_label: float = 2.5) -> float:
    return _parse_label_with_status(text, fallback_label)[0]


def _parse_rgb(value: str, default: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        return default
    try:
        return tuple(max(0, min(255, int(part))) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return default


def _resize_with_padding(
    image: Image.Image,
    target_size: int,
    background_color: tuple[int, int, int],
) -> Image.Image:
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        return image.convert("RGB").resize((target_size, target_size))

    scale = min(target_size / source_width, target_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = image.convert("RGB").resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), color=background_color)
    canvas.paste(resized, ((target_size - resized_width) // 2, (target_size - resized_height) // 2))
    return canvas


def _pil_image_to_data_url(image: Image.Image, image_format: str, jpeg_quality: int) -> str:
    buffered = BytesIO()
    save_kwargs: dict[str, Any] = {}
    normalized_format = image_format.upper()
    if normalized_format == "JPEG":
        image = image.convert("RGB")
        save_kwargs.update(quality=jpeg_quality, optimize=True)
    image.save(buffered, format=normalized_format, **save_kwargs)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = "jpeg" if normalized_format == "JPEG" else "png"
    return f"data:image/{mime};base64,{encoded}"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


class CropVLMScorer:
    def __init__(self, prompt_path: str) -> None:
        load_env_files()
        try:
            from openai import AzureOpenAI
        except ImportError as error:
            raise RuntimeError("VLM reward requires the 'openai' package") from error

        endpoint = os.getenv("GPT5_AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT).strip()
        deployment = os.getenv(
            "CROP_VLM_MODEL",
            os.getenv("GPT5_AZURE_OPENAI_DEPLOYMENT", DEFAULT_AZURE_DEPLOYMENT),
        ).strip()
        api_version = os.getenv("GPT5_AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION).strip()
        api_key = os.getenv("GPT5_AZURE_OPENAI_API_KEY", "").strip()
        managed_identity_client_id = os.getenv(
            "GPT5_AZURE_MANAGED_IDENTITY_CLIENT_ID",
            DEFAULT_MANAGED_IDENTITY_CLIENT_ID,
        ).strip()

        client_kwargs: dict[str, Any] = {
            "api_version": api_version,
            "azure_endpoint": endpoint,
            "max_retries": 0,
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as error:
                raise RuntimeError("managed-identity authentication requires the 'azure-identity' package") from error
            credential = DefaultAzureCredential(managed_identity_client_id=managed_identity_client_id)
            client_kwargs["azure_ad_token_provider"] = _get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )

        self.client = AzureOpenAI(**client_kwargs)
        self.model = deployment
        self.request_timeout = float(os.getenv("CROP_VLM_TIMEOUT", "45"))
        self.max_retries = int(os.getenv("CROP_VLM_MAX_RETRIES", "2"))
        self.retry_backoff = float(os.getenv("CROP_VLM_RETRY_BACKOFF", "1.5"))
        self.fallback_label = float(os.getenv("CROP_VLM_FALLBACK_LABEL", "5.0"))
        self.parse_fallback_label = float(os.getenv("CROP_VLM_PARSE_FALLBACK_LABEL", "2.5"))
        self.eval_image_size = int(os.getenv("CROP_VLM_IMAGE_SIZE", "384"))
        self.image_format = os.getenv("CROP_VLM_IMAGE_FORMAT", "JPEG").upper()
        self.jpeg_quality = int(os.getenv("CROP_VLM_JPEG_QUALITY", "70"))
        self.max_output_tokens = int(os.getenv("CROP_VLM_MAX_OUTPUT_TOKENS", "1024"))
        self.reasoning_effort = os.getenv("CROP_VLM_REASONING_EFFORT", "low").strip().lower()
        self.output_verbosity = os.getenv("CROP_VLM_OUTPUT_VERBOSITY", "low").strip().lower()
        response_log_path = os.getenv("CROP_VLM_LOG_PATH", "").strip()
        self.response_log_path = Path(response_log_path) if response_log_path else None
        visual_log_dir = os.getenv("CROP_VLM_VISUAL_LOG_DIR", "").strip()
        self.visual_log_dir = Path(visual_log_dir) if visual_log_dir else None
        self.visual_log_every = int(os.getenv("CROP_VLM_VISUAL_LOG_EVERY", "0"))
        self._visual_log_counter = 0
        self.preprocess_mode = os.getenv("CROP_VLM_PREPROCESS_MODE", "letterbox").strip().lower()
        self.background_color = _parse_rgb(os.getenv("CROP_VLM_BG_COLOR", "255,255,255"))

        if self.max_retries < 0:
            raise ValueError("CROP_VLM_MAX_RETRIES must be non-negative")
        if self.eval_image_size <= 0:
            raise ValueError("CROP_VLM_IMAGE_SIZE must be positive")
        if self.visual_log_every < 0:
            raise ValueError("CROP_VLM_VISUAL_LOG_EVERY must be non-negative")
        if self.visual_log_every and self.visual_log_dir is None:
            raise ValueError("CROP_VLM_VISUAL_LOG_DIR is required when CROP_VLM_VISUAL_LOG_EVERY is positive")
        if self.image_format not in {"JPEG", "PNG"}:
            raise ValueError("CROP_VLM_IMAGE_FORMAT must be JPEG or PNG")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("CROP_VLM_REASONING_EFFORT must be low, medium, or high")
        if self.output_verbosity not in {"low", "medium", "high"}:
            raise ValueError("CROP_VLM_OUTPUT_VERBOSITY must be low, medium, or high")
        if self.preprocess_mode not in {"letterbox", "stretch"}:
            raise ValueError("CROP_VLM_PREPROCESS_MODE must be letterbox or stretch")

        self.rule_prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
        if not self.rule_prompt:
            raise ValueError("VLM prompt must be non-empty")

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        if self.preprocess_mode == "stretch":
            return image.convert("RGB").resize((self.eval_image_size, self.eval_image_size))
        return _resize_with_padding(image, self.eval_image_size, self.background_color)

    def _save_visual_artifacts(
        self,
        original: Image.Image,
        candidate: Image.Image,
        log_context: dict[str, Any] | None,
    ) -> dict[str, str] | None:
        if self.visual_log_dir is None or self.visual_log_every == 0:
            return None

        self._visual_log_counter += 1
        if self._visual_log_counter % self.visual_log_every != 0:
            return None

        context = log_context or {}
        sample_id = str(context.get("sample_id", "sample"))
        safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sample_id)
        suffix = f"{safe_sample_id}_{os.getpid()}_{self._visual_log_counter:06d}"
        extension = "jpg" if self.image_format == "JPEG" else "png"
        self.visual_log_dir.mkdir(parents=True, exist_ok=True)
        original_path = self.visual_log_dir / f"{suffix}_original.{extension}"
        candidate_path = self.visual_log_dir / f"{suffix}_candidate.{extension}"
        save_kwargs: dict[str, Any] = {"quality": self.jpeg_quality} if self.image_format == "JPEG" else {}
        original.save(original_path, format=self.image_format, **save_kwargs)
        candidate.save(candidate_path, format=self.image_format, **save_kwargs)

        if self.response_log_path is None:
            return {"original": str(original_path), "candidate": str(candidate_path)}
        return {
            "original": Path(os.path.relpath(original_path, self.response_log_path.parent)).as_posix(),
            "candidate": Path(os.path.relpath(candidate_path, self.response_log_path.parent)).as_posix(),
        }

    def score(
        self,
        original: Image.Image,
        candidate: Image.Image,
        caption: str,
        headline: str,
        log_context: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        result = self.score_detailed(original, candidate, caption, headline, log_context)
        return result.reward, result.label

    def score_detailed(
        self,
        original: Image.Image,
        candidate: Image.Image,
        caption: str,
        headline: str,
        log_context: dict[str, Any] | None = None,
    ) -> VLMScoreResult:
        original_prepared = self._prepare_image(original)
        candidate_prepared = self._prepare_image(candidate)
        visual_artifacts = self._save_visual_artifacts(original_prepared, candidate_prepared, log_context)
        original_url = _pil_image_to_data_url(original_prepared, self.image_format, self.jpeg_quality)
        candidate_url = _pil_image_to_data_url(candidate_prepared, self.image_format, self.jpeg_quality)
        user_text = (
            f"{self.rule_prompt}\n\n"
            "Input Fields:\n"
            f"- Image Caption: {caption or '[not provided]'}\n"
            f"- Article Headline: {headline}\n\n"
            "Image A is the original reference. Image B is the candidate thumbnail. "
            "Return only one valid JSON object following the required schema."
        )

        last_error: Exception | None = None
        score_started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                request_started = time.perf_counter()
                with self.client.responses.stream(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Image A (Original Reference)"},
                                {"type": "input_image", "image_url": original_url},
                                {"type": "input_text", "text": "Image B (Thumbnail/Candidate)"},
                                {"type": "input_image", "image_url": candidate_url},
                                {"type": "input_text", "text": user_text},
                            ],
                        }
                    ],
                    reasoning={"effort": self.reasoning_effort},
                    text={"verbosity": self.output_verbosity},
                    max_output_tokens=self.max_output_tokens,
                    timeout=self.request_timeout,
                ) as stream:
                    response = stream.get_final_response()
                request_latency_ms = (time.perf_counter() - request_started) * 1000
                output_text = extract_response_text(response)
                label, parsed = _parse_label_with_status(output_text, self.parse_fallback_label)
                reward = (5.0 - label) / 5.0
                status = "completed" if parsed else "parse_fallback"
                response_id = getattr(response, "id", None)
                latency_ms = (time.perf_counter() - score_started) * 1000
                if self.response_log_path is not None:
                    _append_jsonl(
                        self.response_log_path,
                        {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "status": status,
                            "model": self.model,
                            "response_id": response_id,
                            "attempt": attempt + 1,
                            "request_latency_ms": round(request_latency_ms, 2),
                            "latency_ms": round(latency_ms, 2),
                            "headline": headline,
                            "context": log_context or {},
                            "label": label,
                            "reward": reward,
                            "output_text": output_text,
                            "visual_artifacts": visual_artifacts,
                        },
                    )
                return VLMScoreResult(
                    reward=reward,
                    label=label,
                    status=status,
                    output_text=output_text,
                    response_id=response_id,
                    attempt_count=attempt + 1,
                    latency_ms=latency_ms,
                )
            except Exception as error:  # noqa: BLE001 - SDK and credential failures share the same fallback.
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff**attempt)

        print(
            f"[crop_vlm] request failed after {self.max_retries + 1} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )
        label = max(0.0, min(5.0, self.fallback_label))
        reward = (5.0 - label) / 5.0
        latency_ms = (time.perf_counter() - score_started) * 1000
        error_type = type(last_error).__name__
        if self.response_log_path is not None:
            _append_jsonl(
                self.response_log_path,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "status": "failed",
                    "model": self.model,
                    "headline": headline,
                    "attempts": self.max_retries + 1,
                    "latency_ms": round(latency_ms, 2),
                    "context": log_context or {},
                    "label": label,
                    "reward": reward,
                    "error_type": error_type,
                    "visual_artifacts": visual_artifacts,
                },
            )
        return VLMScoreResult(
            reward=reward,
            label=label,
            status="failed",
            output_text=None,
            response_id=None,
            attempt_count=self.max_retries + 1,
            latency_ms=latency_ms,
            error_type=error_type,
        )


@lru_cache(maxsize=8)
def get_crop_vlm_scorer(prompt_path: str) -> CropVLMScorer:
    return CropVLMScorer(prompt_path)