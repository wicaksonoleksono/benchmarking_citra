import timm
import logging
from typing import Union, List, Dict
import torch.nn as nn

logger = logging.getLogger(__name__)


def init_mobilenetv3_models(num_heads: int, model_name: str):

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_heads
    )
    # (optional) make sure everything’s unfrozen
    for p in model.parameters():
        p.requires_grad = True
    return model
