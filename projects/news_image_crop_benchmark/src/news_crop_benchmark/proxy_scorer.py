from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from news_crop_benchmark.geometry import BBox


@dataclass(frozen=True)
class VisualProxyMetrics:
    saliency: float
    composition: float
    integrity: float
    area: float
    area_fraction: float


def crop_area_score(
    area_fraction: float,
    minimum: float = 0.05,
    preferred_minimum: float = 0.25,
    preferred_maximum: float = 0.75,
    maximum: float = 0.95,
) -> float:
    if not 0.0 <= minimum < preferred_minimum <= preferred_maximum < maximum <= 1.0:
        raise ValueError("area thresholds must be ordered within [0, 1]")
    if area_fraction <= minimum or area_fraction >= maximum:
        return 0.0
    if area_fraction < preferred_minimum:
        return (area_fraction - minimum) / (preferred_minimum - minimum)
    if area_fraction <= preferred_maximum:
        return 1.0
    return (maximum - area_fraction) / (maximum - preferred_maximum)


def crop_image(image: Image.Image, bbox: BBox) -> Image.Image:
    width, height = image.size
    left = max(0, min(width - 1, int(round(bbox.x1))))
    top = max(0, min(height - 1, int(round(bbox.y1))))
    right = max(left + 1, min(width, int(round(bbox.x2))))
    bottom = max(top + 1, min(height, int(round(bbox.y2))))
    return image.crop((left, top, right, bottom))


def compute_visual_proxy_metrics(image: Image.Image, bbox: BBox, maximum_side: int = 512) -> VisualProxyMetrics:
    if maximum_side <= 0:
        raise ValueError("maximum_side must be positive")
    image = image.convert("RGB")
    image_width, image_height = image.size
    area_fraction = bbox.area / (image_width * image_height)

    scale = min(1.0, maximum_side / max(image_width, image_height))
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR).convert("L")
    grayscale = np.asarray(resized, dtype=np.float32) / 255.0
    gradient_y, gradient_x = np.gradient(grayscale)
    energy = np.hypot(gradient_x, gradient_y)

    x1 = max(0, min(resized_width - 1, int(math.floor(bbox.x1 * scale))))
    y1 = max(0, min(resized_height - 1, int(math.floor(bbox.y1 * scale))))
    x2 = max(x1 + 1, min(resized_width, int(math.ceil(bbox.x2 * scale))))
    y2 = max(y1 + 1, min(resized_height, int(math.ceil(bbox.y2 * scale))))
    crop_energy = energy[y1:y2, x1:x2]

    total_energy = float(energy.sum())
    retained_energy = float(crop_energy.sum())
    retention = retained_energy / total_energy if total_energy > 1e-8 else area_fraction
    saliency = float(np.clip(0.5 + 0.5 * (retention - area_fraction), 0.0, 1.0))

    if retained_energy > 1e-8:
        yy, xx = np.indices(crop_energy.shape, dtype=np.float32)
        center_x = float((xx * crop_energy).sum() / retained_energy) / max(1, crop_energy.shape[1] - 1)
        center_y = float((yy * crop_energy).sum() / retained_energy) / max(1, crop_energy.shape[0] - 1)
        center_distance = math.hypot(center_x - 0.5, center_y - 0.5) / math.sqrt(0.5)
        composition = float(np.clip(1.0 - center_distance, 0.0, 1.0))
    else:
        composition = 0.5

    band = max(1, min(crop_energy.shape) // 20)
    boundary_mask = np.zeros(crop_energy.shape, dtype=bool)
    boundary_mask[:band, :] = True
    boundary_mask[-band:, :] = True
    boundary_mask[:, :band] = True
    boundary_mask[:, -band:] = True
    boundary_energy = float(crop_energy[boundary_mask].mean()) if crop_energy.size else 0.0
    global_energy = float(energy.mean())
    integrity = 1.0 / (1.0 + boundary_energy / max(global_energy, 1e-8))

    area = crop_area_score(area_fraction)
    return VisualProxyMetrics(
        saliency=saliency,
        composition=composition,
        integrity=float(np.clip(integrity, 0.0, 1.0)),
        area=area,
        area_fraction=area_fraction,
    )


class ClipTitleScorer:
    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoProcessor, CLIPModel

        self._torch = torch
        self._device = torch.device(device)
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self._model = CLIPModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()
        self._lock = threading.Lock()

    def score(self, title: str, original: Image.Image, candidate: Image.Image) -> tuple[float, float]:
        inputs = self._processor(
            text=[title],
            images=[original.convert("RGB"), candidate.convert("RGB")],
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._lock, self._torch.inference_mode():
            output = self._model(**inputs)
        similarities = (output.image_embeds @ output.text_embeds.T).squeeze(-1).float().cpu().tolist()
        return float(similarities[0]), float(similarities[1])


_CLIP_SCORERS: dict[tuple[str, str], ClipTitleScorer] = {}
_CLIP_SCORERS_LOCK = threading.Lock()


def get_clip_title_scorer(model_path: str, device: str = "cpu") -> ClipTitleScorer:
    key = (model_path, device)
    with _CLIP_SCORERS_LOCK:
        scorer = _CLIP_SCORERS.get(key)
        if scorer is None:
            scorer = ClipTitleScorer(model_path=model_path, device=device)
            _CLIP_SCORERS[key] = scorer
        return scorer


def relative_title_relevance(original_similarity: float, candidate_similarity: float, scale: float = 20.0) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive")
    delta = candidate_similarity - original_similarity
    return 1.0 / (1.0 + math.exp(-scale * delta))
