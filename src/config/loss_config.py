from .base_config import BaseConfig
from typing import Dict, Any


class LossConfig(BaseConfig):
    required_arguments = {"name"}

    def __init__(self, name: str,  arguments: Dict[str, Any]):
        self.name: str = name
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LossConfig":
        cls.check_required(d)
        config = LossConfig(name=d['name'],
                            arguments=d.get("arguments", {}))
        return config
