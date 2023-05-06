from __future__ import annotations
from typing import Dict, Any
from config.base_config import BaseConfig
from config.data_config import DataConfig
from config.model_config import GraphLSTMConfig
from config.train_config import TrainConfig


class CDRConfig(BaseConfig):
    """The CDR Configuration class"""
    # chemical_string = "Chemical"
    # disease_string = "Disease"
    # adjacency_rel = "node"
    # root_rel = "root"

    required_arguments = {"experiment", "experiment_dir", "train", "model",
                          "data"}

    def __init__(self,
                 data: DataConfig,
                 model: GraphLSTMConfig,
                 train: TrainConfig,
                 experiment: str,
                 experiment_dir: str):
        self.data = data
        self.model = model
        self.train = train
        self.experiment = experiment
        self.experiment_dir = experiment_dir

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CDRConfig":
        cls.check_required(d)
        data = DataConfig.from_dict(d['data'])
        model = GraphLSTMConfig.from_dict(d['model'])
        train = TrainConfig.from_dict(d['train'])
        config = CDRConfig(data=data,
                           model=model,
                           train=train,
                           experiment=d['experiment'],
                           experiment_dir=d['experiment_dir'])
        return config
