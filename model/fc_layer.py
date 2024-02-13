import torch.nn as nn
import torch.nn.functional as F
import torch

class FullyConnected(nn.Module):
	"""
	Projection head for contrastive learning.

	Args:
		input_dim: Input dimension of the projection head.
		hidden_dim: Hidden dimension of the projection head.
		output_dim: Output dimenstion of the projection head. 
	"""

	def __init__(self, input_dim=512, num_captions=5):
		super(FullyConnected, self).__init__()

		self.num_captions = num_captions

		torch.manual_seed(42)

		self.linear1 = nn.Linear(input_dim, 1)
		self.relu = nn.ReLU()


	def forward(self, captions):
		"""
		Forward pass through the projection head.

		Args:
			captions: List of input captions
		"""

		weights = []

		for caption in captions:

			caption = self.linear1(caption)
			caption = self.relu(caption)

			weights.append(caption)

		weights = torch.stack(weights).to('cuda:3')
		weights = F.softmax(weights, dim=0)

		weighted_features = captions * weights 

		return weighted_features
