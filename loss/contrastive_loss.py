import torch
import torch.nn.functional as F
import lightning.pytorch as pl

class NTXentLoss(pl.LightningModule):
	def __init__(self, temperature=0.5):
		super(NTXentLoss, self).__init__()
		self.temperature = temperature

	def forward(self, z1, z2):

		z1 = F.normalize(z1, dim=-1, p=2)
		z2 = F.normalize(z2, dim=-1, p=2)

		z = torch.cat([z1, z2], dim=0)

		sim_matrix = torch.matmul(z, z.t()) / self.temperature

		mask = torch.eye(len(z), device=self.device)
		sim_matrix = sim_matrix - mask * 1e9

		labels = torch.arange(0, len(z), device=self.device)
		labels = torch.roll(labels, shifts=len(labels) // 2, dims=0)

		log_prob_matrix = F.log_softmax(sim_matrix, dim=-1)

		loss = -torch.sum(F.one_hot(labels, num_classes=len(z))) * log_prob_matrix / torch.sum(F.one_hot(labels, num_classes=len(z)))

		return loss