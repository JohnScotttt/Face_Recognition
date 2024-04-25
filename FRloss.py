import torch
from torch import nn
import torch.nn.functional as F

class FaceRecognitionLoss(nn.Module):
    def __init__(self):
        super(FaceRecognitionLoss, self).__init__()

    def forward(self, batch, label):
        query = batch[0].view(1, -1)
        key = batch[1:]
        result = nn.CosineSimilarity()(query, key).unsqueeze(0)
        return nn.CrossEntropyLoss()(result, label)