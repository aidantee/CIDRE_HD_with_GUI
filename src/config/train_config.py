from typing import Optional, Dict, Any

from config.base_config import BaseConfig
from config.loss_config import LossConfig
from config.lr_scheduler_config import LRSchedulerConfig
from config.optimizer_config import OptimizerConfig


class TrainConfig(BaseConfig):
    required_arguments = {"optimizer", "loss",
                          "batch_size", "batch_max_tokens", "eval_interval", "log_interval",
                          "num_epochs", "gradient_clipping"}

    def __init__(self,
                 optimizer: OptimizerConfig,
                 loss: LossConfig,
                 lr_scheduler: Optional[LRSchedulerConfig],
                 batch_size: int,
                 batch_max_tokens: int,
                 eval_interval: int,
                 log_interval: int,
                 num_epochs: int,
                 gradient_clipping: int):
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.batch_max_tokens = batch_max_tokens
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.num_epochs = num_epochs
        self.gradient_clipping = gradient_clipping

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        cls.check_required(d)
        optimizer = OptimizerConfig.from_dict(d['optimizer'])
        loss = LossConfig.from_dict(d['loss'])
        lr_scheduler = LRSchedulerConfig.from_dict(d['lr_scheduler']) if 'lr_scheduler' in d else None

        config = TrainConfig(
            optimizer=optimizer,
            loss=loss,
            lr_scheduler=lr_scheduler,
            batch_size=d['batch_size'],
            batch_max_tokens=d['batch_max_tokens'],
            eval_interval=d['eval_interval'],
            log_interval=d['log_interval'],
            num_epochs=d['num_epochs'],
            gradient_clipping=d['gradient_clipping'],
        )
        return config
