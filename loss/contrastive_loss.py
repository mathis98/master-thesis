import torch
import torch.nn as nn
import torch.nn.functional as F


def device_as(t1, t2):
	"""
	Move the input tensor to the same device as the target tensor.

	Args:
		t1 (torch.Tensor): Input tensor.
		t2 (torch.Tensor): Target tensor.

	Returns:
		torch.Tensor: t1 moved to the same device as t2.
	"""

	return t1.to(t2.device)

class SimCLRLoss(nn.Module):
	"""
	Contrastive loss for SimCRL (NT-Xent).

	Args:
		temperature (float): Temperature for scaling the logits.
	"""

	def __init__(self,temperature=.07):
		super(SimCLRLoss, self).__init__()
		self.temperature = temperature

	def calc_similarity_batch(self, a, b):
		"""
		Calculate cosine similarity between representations of batch a and batch b.

		Args:
			a (torch.Tensor): Input tensor a.
			b (torch.Tensor): Input tensor b.

		Returns:
			torch.Tensor: Cosine similarity matrix between representation of batch a and batch b.
		"""

		representations = torch.cat([a, b], dim=0)
		return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

	def update_temperature(self, current_batch_size):
		self.temperature = self.temperature / current_batch_size

	def forward(self, z_i, z_j):
		"""
		Forward pass to calculate the NT-Xent loss.

		Args:
			z_i (torch.Tensor): Embeddings from first batch (Image).
			z_j (torch.Tensor): Embeddings from the second batch (Caption).

		Returns:
			torch.Tensor: NT-Xent loss.
		"""

		batch_size = z_i.shape[0]

		mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

		z_i = F.normalize(z_i, p=2, dim=1)
		z_j = F.normalize(z_j, p=2, dim=1)

		similarity_matrix = self.calc_similarity_batch(z_i, z_j)

		sim_ij = torch.diag(similarity_matrix, batch_size)
		sim_ji = torch.diag(similarity_matrix, -batch_size)

		positives = torch.cat([sim_ij, sim_ji], dim=0)

		self.update_temperature(z_i.shape[0])

		nominator = torch.exp(positives / self.temperature)

		denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

		num_valid_pairs = torch.sum(mask) / 2

		all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
		loss = torch.sum(all_losses) / (2 * batch_size)
		
		return loss