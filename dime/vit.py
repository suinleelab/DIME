import torch
import torch.nn as nn

class PredictorViT(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super(PredictorViT, self).__init__()
        self.fc = nn.Linear(backbone.embed_dim, num_classes)
        self.backbone = backbone
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.backbone.forward_head(x, pre_logits=True)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SelectorViT(nn.Module):
    def __init__(self, backbone):
        super(SelectorViT, self).__init__()
        self.backbone = backbone
        self.hidden1 = nn.Linear(backbone.embed_dim, 1)

    def forward(self, x):
        x = self.backbone.forward_features(x)[:, 1:]
        x = self.hidden1(x).squeeze()

        return x

class ValueNetworViT(nn.Module):
    def __init__(self, backbone, mask_width=7, dropout=0.3, use_entropy=True):
        super(ValueNetworViT, self).__init__()
        self.backbone = backbone
        self.hidden1 = nn.Linear(backbone.embed_dim, 1)
        self.flatten = nn.Flatten()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.use_entropy = use_entropy

    
    def forward(self, x):
        x = self.backbone.forward_features(x)[:, 1:]
        x = self.hidden1(x).squeeze()
        if self.use_entropy:
            x = self.sigmoid(x)
        else:
            x = self.softplus(x)
        
        return x

class PredictorSemiSupervisedVit(nn.Module):
    def __init__(self, backbone1, backbone2, num_classes=10):
        super(PredictorSemiSupervisedVit, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.fc = nn.Linear(backbone1.embed_dim * 2, num_classes)
    def forward(self, x, x_sketch):
        x = self.backbone1.forward_features(x)
        x = self.backbone1.forward_head(x, pre_logits=True)
        
        x_sketch = self.backbone2.forward_features(x_sketch)
        x_sketch = self.backbone2.forward_head(x_sketch, pre_logits=True)
        
        x_cat = torch.cat((x, x_sketch), dim=1)
        # x = x.view(x.size(0), -1)
        x_cat = self.fc(x_cat)
        return x_cat


class ValueNetworkSemiSupervisedVit(nn.Module):
    def __init__(self, backbone1, backbone2, hidden=512, dropout=0.3, use_entropy=True):
        super(ValueNetworkSemiSupervisedVit, self).__init__()
        self.dropout = dropout
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.linear1 = nn.Linear(backbone1.embed_dim + backbone2.embed_dim, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.use_entropy = use_entropy
    
    def forward(self, x, x_sketch):
        x = self.backbone1.forward_features(x)[:, 1:]
        x_sketch = self.backbone2.forward_features(x_sketch)[:, 1:]
        x_cat = torch.cat((x, x_sketch), dim=2)
        x_cat = self.dropout(self.relu(self.linear1(x_cat)))
        x_cat = self.dropout(self.relu(self.linear2(x_cat)))
        x_cat = self.linear3(x_cat).squeeze()
        if self.use_entropy:
            x_cat = self.sigmoid(x_cat)
        else:
            x_cat = self.softplus(x_cat)

        return x_cat




