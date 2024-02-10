import torch.nn as nn
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

		batch_size = captions.size(0)
		weights = torch.zeros(batch_size, self.num_captions, 1, device=captions.device)

		for i in range(self.num_captions):
			caption_features = captions[:,i,:]
			weights[:,i,:] = self.relu(self.linear1(caption_features))

		weights = F.softmax(weights, dim=1)
		weighted_features = captions * weights 
		aggregated_features = torch.mean(weighted_features, dim=1)

		return aggregated_features
