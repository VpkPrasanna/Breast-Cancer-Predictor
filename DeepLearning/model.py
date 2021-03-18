import torch.nn as nn
import timm
target_size = 2


class BreastTumorModel(nn.Module):
    def __init__(self,model_name="tf_efficientnet_b4_ns",pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name,pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features,target_size)
    
    def forward(self,x):
        x = self.model(x)
        return x
