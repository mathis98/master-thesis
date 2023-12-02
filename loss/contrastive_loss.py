import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class NTXentLoss(pl.LightningModule):
	def __init__(self, temperature=0.5):
		super(NTXentLoss, self).__init__()
		self.temperature = temperature

	def forward(self, z1, z2):
		# z1 and z2 are the embeddings from two augmented views of the same image
		# Make sure z1 and z2 have the same shape

		# Normalize the embeddings
		z1 = F.normalize(z1, dim=-1, p=2)
		z2 = F.normalize(z2, dim=-1, p=2)

		# Concatenate the embeddings
		z = torch.cat([z1, z2], dim=0)

		# Compute pairwise cosine similarity
		sim_matrix = torch.matmul(z, z.t()) / self.temperature

		# Set diagonal elements to a large negative value to avoid them being the maximum
		mask = torch.eye(len(z), device=self.device)
		sim_matrix = sim_matrix - mask * 1e9

		# Create target labels for positive and negative pairs
		labels = torch.arange(0, len(z), device=self.device)
		labels = torch.roll(labels, shifts=len(labels) // 2, dims=0)

		# Compute the log probability for each sample
		log_prob_matrix = F.log_softmax(sim_matrix, dim=-1)

		# Compute the loss
		loss = F.cross_entropy(log_prob_matrix, labels)

		return loss

# Example usage:
# Instantiate the loss function
ntxent_loss = NTXentLoss(temperature=0.5)

# Generate two batches of embeddings (z1 and z2)
z1 = torch.randn((batch_size, embedding_size))
z2 = torch.randn((batch_size, embedding_size))

# Compute the contrastive loss
loss = ntxent_loss(z1, z2)
