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

			print(f'caption: {caption}')

			caption = self.linear1(caption)

			print(f'after linear: {caption}')

			caption = self.relu(caption)

			print(f'after relu: {caption}')

			weights.append(caption)

		weights = torch.stack(weights).to('cuda:3')

		print(f'weights of captions: {weights}')

		weights = F.softmax(weights)

		print(f'softmaxed: {weights}')

		weighted_features = captions * weights 

		# weighted_features = weighted_features.squeeze(0)

		print(f'weighted features: {weighted_features}')

		return weighted_features
