import torch.nn as nn

def cross_entropy_loss(labels, outputs):
    loss = nn.CrossEntropyLoss(outputs, labels)

    return loss