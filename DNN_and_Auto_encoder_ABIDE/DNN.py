import torch.nn as nn
import torch


###################################################################################################################
###################################################################################################################
###################################################################################################################
def single_linear(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=out_channels),
        nn.Dropout(0.7),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(out_channels)
    )

class DNN_2D_1_dropout(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_1 = single_linear(5000, 500)
        self.linear_6 = nn.Linear(in_features=500, out_features=30)
        self.linear_7 = nn.Linear(in_features=30, out_features=2)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        input = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.linear_1(input)
        x = self.linear_6(x)
        x = self.dropout(x)
        x = self.linear_7(x)
        out = self.softmax(x)



        return out




