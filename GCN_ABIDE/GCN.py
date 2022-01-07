import torch.nn as nn
import torch
import torch_geometric as tg



class ChebNet_one_dropout(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K, dropout_rate, normalize=True):

        super(ChebNet_one_dropout, self).__init__()
        self.normalize = normalize

        self.conv1 = tg.nn.ChebConv(in_c, hid_c, K, normalization='sym', bias=True)
        self.conv2 = tg.nn.ChebConv(hid_c, hid_c, K, normalization='sym', bias=True)
        self.conv3 = tg.nn.ChebConv(hid_c, out_c, K, normalization='sym', bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, data, edge_index, edgenet_input):

        edge_weight = torch.squeeze(edgenet_input)
        data = self.dropout(data)
        h = self.relu(self.conv1(data, edge_index, edge_weight))
        h = self.relu(self.conv2(h, edge_index, edge_weight))
        h = self.conv3(h, edge_index, edge_weight)

        return self.softmax(h)


















