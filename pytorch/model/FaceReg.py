import timm
import torch
import torch.nn as nn
import torch.nn.functional as F  # Corrected import
import math
model_names = "mnasnet_small.lamb_in1k"


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features, labels):
        # Normalize input features and the class-center weights
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create a one-hot vector for the labels
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Set the logit for the correct class to phi
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Compute the final cross-entropy loss
        return F.cross_entropy(output, labels)

    # --- ADD THIS METHOD ---
    def get_logits(self, input_features):
        """
        Calculates the logits for metric computation without applying the margin.
        This should be used within a torch.no_grad() context.
        """
        # Just calculates the cosine similarity, which acts as logits for prediction
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        return self.s * cosine  # Return scaled logits


class FM(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=False):
        super(FM, self).__init__()
        self.model_name = model_name  # <-- This line is the key
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )
        feature_size = self.backbone.num_features
        self.head = ArcFaceLoss(in_features=feature_size, out_features=num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, images, labels):
        features = self.backbone(images)
        loss = self.head(features, labels)
        return loss

    def get_logits(self, images):

        with torch.no_grad():
            features = self.backbone(images)
            # Call the get_logits method from the ArcFaceLoss head
            logits = self.head.get_logits(features)
            return logits

    def inference(self, images):
        # The inference method returns the final feature embedding
        with torch.no_grad():
            features = self.backbone(images)
            return F.normalize(features)
