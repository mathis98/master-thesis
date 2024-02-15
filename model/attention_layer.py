import torch.nn as nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
	"""
	Transformer Encoder based layer for weight generation
	"""

	def __init__(self, input_dim=512, num_layers=2, num_heads=4, hidden_dim=256):
		super(Attention, self).__init__()
		encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
		self.attention = nn.TranformerEncoder(encoder_layer, num_layers=num_layers)

		self.linear = nn.Linear(input_dim, 1)
		self.relu = nn.ReLU()

	def forward(self, captions):
		"""
		Forward pass through the attention layers

		Args:
			captions: List of input captions
		"""

		weights = []

		for caption in captions:

			transformer_encode = self.attention(caption)

			weight = self.linear(transformer_encode)
			weight = self.relu(weight)

			weights.append(weight)

		weights = torch.stack(weights).to('cuda:3')

		weights = F.softmax(weights, dim=0)

		weighted_features = captions * weights 


		return weighted_features
