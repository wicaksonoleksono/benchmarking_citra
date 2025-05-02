import timm
import logging
import torch.nn as nn
logger = logging.getLogger(__name__)


def init_model(num_heads: int, model_name: str):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_heads
    )
    for p in model.parameters():
        p.requires_grad = True
    return model
