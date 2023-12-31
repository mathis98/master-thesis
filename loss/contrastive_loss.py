import torch
import torch.nn as nn
import torch.nn.functional as F


def device_as(t1, t2):
	return t1.to(t2.device)

class SimCLRLoss(nn.Module):
	def __init__(self,temperature=.07):
		super(SimCLRLoss, self).__init__()
		self.temperature = temperature

	def calc_similarity_batch(self, a, b):
		representations = torch.cat([a, b], dim=0)
		return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

	def forward(self, z_i, z_j):
		batch_size = z_i.shape[0]

		mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

		z_i = F.normalize(z_i, p=2, dim=1)
		z_j = F.normalize(z_j, p=2, dim=1)

		similarity_matrix = self.calc_similarity_batch(z_i, z_j)

		sim_ij = torch.diag(similarity_matrix, batch_size)
		sim_ji = torch.diag(similarity_matrix, -batch_size)

		positives = torch.cat([sim_ij, sim_ji], dim=0)

		nominator = torch.exp(positives / self.temperature)

		denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

		all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
		loss = torch.sum(all_losses) / (2 * batch_size)
		
		return loss