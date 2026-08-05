from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RewardWeights:
    title_relevance: float = 0.45
    saliency: float = 0.20
    composition: float = 0.15
    integrity: float = 0.10
    area: float = 0.10


@dataclass(frozen=True)
class RewardComponents:
    valid: bool
    title_relevance: float = 0.0
    saliency: float = 0.0
    composition: float = 0.0
    integrity: float = 0.0
    area: float = 0.0

    def validate(self) -> None:
        for field in fields(self):
            if field.name == "valid":
                continue
            value = getattr(self, field.name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field.name} must be in [0, 1]")


def combine_proxy_reward(
    components: RewardComponents,
    weights: RewardWeights = RewardWeights(),
    invalid_reward: float = -1.0,
) -> float:
    """Combine proxy metrics without letting visual quality mask title regression."""
    components.validate()
    if not components.valid:
        return invalid_reward

    positive_score = (
        weights.title_relevance * components.title_relevance
        + weights.saliency * components.saliency
        + weights.composition * components.composition
        + weights.integrity * components.integrity
        + weights.area * components.area
    )
    if components.title_relevance < 0.5:
        positive_score = min(positive_score, components.title_relevance)
    return positive_score
