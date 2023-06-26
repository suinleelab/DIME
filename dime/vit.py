import torch
import torch.nn as nn


class PredictorViT(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(backbone.embed_dim, num_classes)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.backbone.forward_head(x, pre_logits=True)
        x = self.fc(x)
        return x


class ValueNetworkViT(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.hidden = nn.Linear(backbone.embed_dim, 1)

    def forward(self, x):
        x = self.backbone.forward_features(x)[:, 1:]
        x = self.hidden(x).squeeze()
        return x


class PredictorViTPrior(nn.Module):
    def __init__(self, backbone1, backbone2, num_classes=10):
        super().__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.fc = nn.Linear(backbone1.embed_dim * 2, num_classes)

    def forward(self, x, prior):
        x = self.backbone1.forward_features(x)
        x = self.backbone1.forward_head(x, pre_logits=True)

        prior = self.backbone2.forward_features(prior)
        prior = self.backbone2.forward_head(prior, pre_logits=True)

        x_cat = torch.cat((x, prior), dim=1)
        x_cat = self.fc(x_cat)
        return x_cat


# TODO why does this one get so many fc layers but the one above gets only one?
class ValueNetworkViTPrior(nn.Module):
    def __init__(self, backbone1, backbone2, hidden=512, dropout=0.3, use_entropy=True):
        super().__init__()
        self.dropout = dropout
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(backbone1.embed_dim + backbone2.embed_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1)
        self.use_entropy = use_entropy

    def forward(self, x, prior):
        x = self.backbone1.forward_features(x)[:, 1:]
        prior = self.backbone2.forward_features(prior)[:, 1:]
        x_cat = torch.cat((x, prior), dim=2)
        x_cat = self.dropout(self.linear1(x_cat).relu())
        x_cat = self.dropout(self.linear2(x_cat).relu())
        x_cat = self.linear3(x_cat).squeeze()
        # TODO should delete this
        if self.use_entropy:
            x_cat = x_cat.sigmoid()
        else:
            x_cat = nn.functional.softplus(x_cat)

        return x_cat
