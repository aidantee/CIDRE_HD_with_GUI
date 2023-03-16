from .base_config import BaseConfig
from typing import Dict, Any


class LRSchedulerConfig(BaseConfig):
    required_arguments = {"name", "arguments"}

    def __init__(self,
                 name: str,
                 arguments: Dict[str, Any]):
        self.name: str = name
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LRSchedulerConfig":
        cls.check_required(d)
        config = LRSchedulerConfig(
            name=d["name"],
            arguments=d.get("arguments", {}))
        return config
