import torch.nn as nn
import torch



class Stacked_autoencoder(nn.Module):
	def __init__(self, in_c, hid_1, hid_2, hid_3, out, dropout_rate):

		super(Stacked_autoencoder, self).__init__()
		self.linear_1 = nn.Linear(in_features=in_c, out_features=hid_1)
		self.linear_2 = nn.Linear(in_features=hid_1, out_features=hid_2)
		self.linear_3 = nn.Linear(in_features=hid_2, out_features=hid_3)
		self.linear_4 = nn.Linear(in_features=hid_3, out_features=out)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, data):

		input = torch.flatten(data, start_dim=1, end_dim=-1)
		lin1 = self.relu(self.linear_1(input))
		lin2 = self.relu(self.dropout(self.linear_2(lin1)))
		lin3 = self.relu(self.linear_3(lin2))
		lin4 = self.relu(self.linear_4(lin3))

		return lin4


class MLP(nn.Module):
	def __init__(self, in_c, hid_1, hid_2, out, dropout_rate):

		super(MLP, self).__init__()
		self.linear_1 = nn.Linear(in_features=in_c, out_features=hid_1)
		self.linear_2 = nn.Linear(in_features=hid_1, out_features=hid_2)
		self.linear_3 = nn.Linear(in_features=hid_2, out_features=out)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, data):

		lin1 = self.relu(self.linear_1(data))
		lin2 = self.dropout(self.relu(self.linear_2(lin1)))
		lin3 = self.relu(self.linear_3(lin2))

		return lin3


class Auto_encoder_MLP(nn.Module):
	def __init__(self, in_c, auto_1, auto_2, auto_3, MLP_1, MLP_2, MLP_out, dropout_rate):

		super(Auto_encoder_MLP, self).__init__()

		self.auto_encoder = Stacked_autoencoder(in_c, auto_1, auto_2, auto_3, in_c, dropout_rate)
		self.MLP = MLP(in_c, MLP_1, MLP_2, MLP_out, dropout_rate)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, data):

		auto_output = self.auto_encoder(data)
		MLP_out = self.MLP(auto_output)
		out = self.softmax(MLP_out)

		return out



