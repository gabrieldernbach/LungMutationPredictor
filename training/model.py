import torch
from torch import nn as nn


class ShallowAggregation(nn.Module):
    def __init__(self, ins, hidden, outs):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(ins, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(ins, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(1024, outs),
        )

    def forward(self, x):  # 8000 x 512
        b, p, d = x.shape
        a_ = self.attn(x.reshape(b * p, d))
        a = nn.functional.softmax(a_.reshape(b, p, 1), dim=1)
        h = (x * a).sum(1)
        return self.mlp(h)


def get_learner():
    model = ShallowAggregation(ins=1024, hidden=2048, outs=1)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=3.125e-05,
        weight_decay=1e-5
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    return model, optimizer, criterion
