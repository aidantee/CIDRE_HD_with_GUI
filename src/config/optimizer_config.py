from .base_config import BaseConfig
from typing import Dict, Any


class OptimizerConfig(BaseConfig):
    required_arguments = {"name", "lr", "weight_decay"}

    def __init__(self, name: str, lr: float, weight_decay: float):
        self.name: str = name
        self.lr: float = lr
        self.weight_decay: float = weight_decay

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        cls.check_required(d)
        config = OptimizerConfig(
            name=d['name'],
            lr=d['lr'],
            weight_decay=d['weight_decay']
        )
        return config
