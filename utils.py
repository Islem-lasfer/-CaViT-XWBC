"""Fonctions utilitaires : graine, scheduler, poids de classes."""

import random

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Warm-up linéaire puis décroissance cosinus (pas par itération)."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_class_weights(train_ds):
    labels = train_ds.df["class"].astype(str).map(train_ds.label_mapping).values
    classes = np.arange(train_ds.num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)
