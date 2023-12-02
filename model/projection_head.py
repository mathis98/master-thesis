import torch.nn as nn

class MyProjectionhead(nn.Module):
	"""
	Projection head for contrastive learning.

	Args:
		input_dim: Input dimension of the projection head.
		hidden_dim: Hidden dimension of the projection head.
		output_dim: Output dimenstion of the projection head. 
	"""

	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MyProjectionhead, self).__init__()

		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.9)
		self.relu = nn.ReLU()
		self.linear2 = nn.Linear(hidden_dim, output_dim)
		self.bn2 = nn.BatchNorm1d(output_dim, momentum=0.9)

	def forward(self, x):
		"""
		Forward pass through the projection head.

		Args:
			x: Input tensor.

		Returns:
			torch.Tensor: Input tensor embedded by the projection head.
		"""

		x = self.linear1(x)
		# x = self.bn1(x)
		x = self.relu(x)
		x = self.linear2(x)
		# x = self.bn2(x)

		return x