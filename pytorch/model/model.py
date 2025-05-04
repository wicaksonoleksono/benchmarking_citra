import timm
import torch.nn as nn


def init_model(
    num_heads: int,
    model_name: str,
    freeze_backbone: bool = False
):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_heads
    )
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        try:
            head = model.get_classifier()
        except AttributeError:
            head = model.classifier
        for param in head.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    return model
