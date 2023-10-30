import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
	def __init__(self, temperature=.07):
		super(SimCLRLoss, self).__init__()
		self.temperature = temperature

	def forward(self, z_i, z_j):
		z_i = F.normalize(z_i, dim=1, p=2)
		z_j = F.normalize(z_j, dim=1, p=2)
		similarity = torch.matmul(z_i, z_j.T) / self.temperature

		positive_pairs = torch.diag(similarity, diagonal=1)

		logits = torch.cat([positive_pairs, positive_pairs], dim=0)

		logits = logits - torch.diag(logits, diagonal=0)

		loss = -torch.log(torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True))

		return loss.mean()