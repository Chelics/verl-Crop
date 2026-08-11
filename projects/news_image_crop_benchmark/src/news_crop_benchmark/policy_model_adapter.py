from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_FAMILIES = ("qwen35", "internvl2", "molmo")


@dataclass(frozen=True)
class PolicyModelAdapter:
    family: str
    trust_remote_code: bool

    @classmethod
    def create(cls, family: str) -> PolicyModelAdapter:
        if family not in MODEL_FAMILIES:
            raise ValueError(f"model family must be one of {MODEL_FAMILIES}: {family}")
        return cls(family=family, trust_remote_code=family != "qwen35")

    def load_renderer(self, model_path: Path) -> Any:
        if self.family == "qwen35":
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(model_path, local_files_only=True)
        if self.family == "internvl2":
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
        return None

    def build_request(
        self,
        renderer: Any,
        prompt: str,
        image: Image.Image,
        *,
        image_max_pixels: int,
        image_min_pixels: int,
    ) -> dict[str, Any]:
        if self.family == "qwen35":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt.replace("<image>\n", "", 1)},
                    ],
                }
            ]
            rendered_prompt = renderer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return {
                "prompt": rendered_prompt,
                "multi_modal_data": {"image": image},
                "mm_processor_kwargs": {
                    "size": {"longest_edge": image_max_pixels, "shortest_edge": image_min_pixels}
                },
            }

        if self.family == "internvl2":
            messages = [{"role": "user", "content": prompt}]
            rendered_prompt = renderer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {"prompt": rendered_prompt, "multi_modal_data": {"image": image}}

        question = prompt.replace("<image>\n", "", 1)
        return {"prompt": question, "multi_modal_data": {"image": image}}

    def llm_kwargs(self, *, internvl_max_dynamic_patch: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.family == "internvl2":
            kwargs["mm_processor_kwargs"] = {"max_dynamic_patch": internvl_max_dynamic_patch}
        return kwargs

    def sampling_kwargs(self, renderer: Any) -> dict[str, Any]:
        if self.family != "internvl2":
            return {}
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [renderer.convert_tokens_to_ids(token) for token in stop_tokens]
        stop_token_ids = [token_id for token_id in stop_token_ids if isinstance(token_id, int) and token_id >= 0]
        return {"stop_token_ids": sorted(set(stop_token_ids))}

    def canonicalize_response(self, response: str) -> tuple[str, bool]:
        stripped = response.strip()
        if stripped.startswith("<crop>"):
            return response, False
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return response, False
        if not isinstance(payload, dict) or set(payload) != {"cx", "cy", "area"}:
            return response, False
        return f"<crop>{stripped}</crop>", True

    def runtime_metadata(self, renderer: Any) -> dict[str, Any]:
        return {
            "model_family": self.family,
            "trust_remote_code": self.trust_remote_code,
            "renderer_class": type(renderer).__name__ if renderer is not None else None,
            "request_format": {
                "qwen35": "processor_chat_template",
                "internvl2": "internlm2_tokenizer_chat_template",
                "molmo": "vllm_molmo_native_text",
            }[self.family],
        }