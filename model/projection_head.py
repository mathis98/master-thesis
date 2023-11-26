import torch.nn as nn

class MyProjectionhead(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MyProjectionhead, self).__init__()

		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.bn1 = nn.BatchNorm1d(hidden_dim)
		self.relu = nn.ReLU()
		self.linear2 = nn.linear(hidden_dim, output_dim)
		self.bn2 = nn.BatchNorm1d(output_dim)

	def forward(self, x):
		x = self.linear1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.linear2(x)
		x = self.bn2(x)

		return x