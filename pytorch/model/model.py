import timm
import torch
import torch.nn as nn


def init_model(num_heads: int, model_name: str, freeze_backbone: bool):
    if model_name == "proxyless_nas":
        target = "proxyless_gpu"
        model = torch.hub.load(
            'mit-han-lab/ProxylessNAS',
            target,
            pretrained=True
        )
        try:
            old_head = model.get_classifier()
            in_feat = old_head.in_features
            new_head = nn.Linear(in_feat, num_heads)
            model.set_classifier(new_head)
        except Exception:
            if hasattr(model, 'classifier'):
                in_feat = model.classifier.in_features
                model.classifier = nn.Linear(in_feat, num_heads)
            elif hasattr(model, 'fc'):
                in_feat = model.fc.in_features
                model.fc = nn.Linear(in_feat, num_heads)
            else:
                raise RuntimeError("Couldn't locate a classifier to replace in ProxylessNAS")
        if freeze_backbone:
            for name, param in model.named_parameters():
                # leave head parameters trainable
                if 'classifier' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        return model
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_heads
    )
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze just the classifier
        try:
            head = model.get_classifier()
            for p in head.parameters():
                p.requires_grad = True
        except AttributeError:
            for p in model.classifier.parameters():
                p.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model
