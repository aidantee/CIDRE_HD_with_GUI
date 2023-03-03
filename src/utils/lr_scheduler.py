import math
from torch import optim


def get_cosine_warm_up_lr_scheduler(optimizer, num_training_steps, warmup_steps):
    if isinstance(warmup_steps, float):
        warmup_steps = num_training_steps * warmup_steps

    def _cosine(step: int) -> float:
        if step < warmup_steps:
            return step / max(1.0, warmup_steps)
        frac = (step - warmup_steps) / max(1.0, num_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * frac)))
    return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                       lr_lambda=_cosine)
