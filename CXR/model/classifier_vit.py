import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MedClipVIT import MedCLIPVisionModelViT


class VIT(nn.Module):
    def __init__(self, cfg, num_labels=4, feature_dim=512):
        super(VIT, self).__init__()
        self.cfg = cfg
        #self._init_classifier()
        #self._init_bn()

        # Use convnext_base as the backbone, but exclude the head
        self.backbone = MedCLIPVisionModelViT(checkpoint=None)
        # Confounder classifier
        self.cls = nn.Sequential(
            nn.Linear(feature_dim, int(feature_dim/4)),
            nn.ReLU()
        )

        self.conf_classifiers = nn.ModuleList()
        for idx, num_class in enumerate(self.cfg.num_conf):
            # Create a separate linear layer for each task
            fc = nn.Linear(int(feature_dim/4), num_class)
            self.conf_classifiers.append(fc)

        # Define one fully connected layer per binary label
        for index in range(num_labels):
            setattr(self, "fc_" + str(index), nn.Linear(feature_dim, 1))  # Binary classification (output size = 1)
    
   
    def forward(self, x):
        # Pass input through the backbone to extract features
        x, early_layer = self.backbone(x, return_layer=3)  # Feature extraction (output shape: [N, feature_dim])
        early_layer = self.cls(early_layer)
        early_layer = early_layer.mean(dim=1) 

        logits = []
        conf_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_conf]

        # Dynamically access classifiers and compute logits for each label
        for index in range(len(self.cfg.num_classes)):  # num_classes should be [1, 1, 1, 1] for binary labels
            x = F.dropout(x, p=self.cfg.fc_drop, training=self.training)
            classifier = getattr(self, "fc_" + str(index))  # Get the corresponding classifier
            logit = classifier(x)  # Apply the classifier to the features (logit shape: (N, 1))
            # print(f"fc logit:{logit.shape}")
            logits.append(logit)


        for index, conf_classifier in enumerate(self.conf_classifiers):
            # Forward pass
            conf_logit = conf_classifier(early_layer)  # shape: (batch_size, num_class)
            # logits = logits.squeeze(dim=1)  # shape: (batch_size,)
            conf_logits[index] = conf_logit


        # Concatenate logits along the second dimension to get shape (N, num_labels)
        # logits = torch.cat(logits, dim=1)  # Shape: (N, num_labels), e.g., (N, 4) for 4 binary labels

        return logits, conf_logits