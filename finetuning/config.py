import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FalConfig:
    key: str


def load_fal_config() -> FalConfig:
    key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not key:
        raise ValueError("Missing FAL_KEY (or FAL_API_KEY) in environment")
    return FalConfig(key=key)
