import torch
from torch import nn
import torch.nn.functional as F
import math

class FR11NmmCELoss(nn.Module):
    def __init__(self, temperature, **kwargs):
        super(FR11NmmCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, batch, label):
        batch = batch / self.temperature
        query = batch[0].view(1, -1)
        key = batch[1:]
        logits = torch.matmul(query, key.T)
        return nn.CrossEntropyLoss()(logits, label)
    
class FR11NArcCELoss(nn.Module):
    def __init__(self, s, margin, **kwargs):
        super(FR11NArcCELoss, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, batch, label):
        query = batch[0].view(1, -1)
        key = batch[1:]
        cosine = torch.matmul(F.normalize(query), F.normalize(key.T))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = cosine * 1.0
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        logits = output * self.s
        return nn.CrossEntropyLoss()(logits, label)

def get_loss_fn(**kwargs):
    if kwargs['name'] == "FR11NmmCELoss":
        return FR11NmmCELoss(**kwargs)
    elif kwargs['name'] == "FR11NArcCELoss":
        return FR11NArcCELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss name {kwargs['name']}")