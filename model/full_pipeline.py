import sys
sys.path.append('..')

import lightning.pytorch as pl
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Embedding for text
from model.text_embedding import BERTSentenceEmbedding

# Embedding for image
from model.image_embedding import ImageEmbeddingModule

# Projection Head
from model.projection_head import MyProjectionhead

# SimCLR loss
from loss.contrastive_loss import SimCLRLoss
# from lightly.loss import NTXentLoss

# for mAP calculation
from utility.helpers import relevant_list, calculate_mAP, define_param_groups, to_cuda_recursive, calculate_bleu


class FullPipeline(pl.LightningModule):
	"""
	Full Pipeline using Pytorch Lightning with a Text classifier, image classifier, and projection head

	Args:
		batch_size (int): batch size for train, val, and test.
		intra (bool): whether to use intra modal loss.
		temperature (float): ntxent temperature.
		learning_rate (float): AdamW learning rate.
		weight_decay (float): AdamW weight decay (L2 regularization).
		max_epochs (int): maximum epochs to train for.
		hidden_dim (int): dimension of embedding.

	Attributes:
		batch_size (int): batch size for train, val, and test.
		intra (bool): whether to use intra modal loss.
		temperature (float): ntxent temperature.
		learning_rate (float): AdamW learning rate.
		weight_decay (float): AdamW weight decay (L2 regularization).
		hidden_dim (int): dimension of embedding.
		resnet_embedding_module (model.image_embedding.ImageEmbeddingModule): Image embedding module.
		bert_embedding_module (model.text_embedding.BERTSentenceEmbeddingModule): Caption embedding module.
		projection_head (model.projection_head.MyProjectionHead): Projection head to shared embedding space.
		criterion (loss.contrastive_loss.SimCLRLoss): NT-Xent loss function.
		max_epochs (int): Maximum epochs to train for.
		validation_step_outputs (list): List to store mAP values during validation.
		test_step_outputs (list): List to store mAP values during testing.
		val_dataloader (DataLoader): Dataloader for validation set.
		test_dataloader (DataLoader): Dataloader for test set.
		top_k (int): map@k
		num_repeats: number of repeats for images (for Repeat technique)
		dataset: Dataset used (UCM or NWPU). Default: UCM
		technique: Technique to use (Repeat, Random, Concat, Mean). Default: Repeat.

	Methods:
		forward(batch): Forward pass through the model.
		training_step(batch, batch_idx): Training step.
		shared_step(batch): Shared step for both validation and testing.
		test_step(batch, batch_idx): Test step.
		on_test_epoch_end(): Called at the end of the test epoch to calculate and log avg mAP.
		validation_step(batch, batch_idx): Validation step.
		on_validation_epoch_end(): Called at the end of the validation epoch to calculate and log avg mAP.
		configure_optimizers(): Configure the optimizer.
		val_dataloader (DataLoader): Dataloader for validation set.
		test_dataloader (DataLoader): Dataloader for test set.
		validation_labels (Tensor): list of labels of validation images.
		test_labels (Tensor): list of labels of test images.
	"""
	def __init__(
		self, 
		val_dataloader=None, 
		test_dataloader=None, 
		batch_size=128, 
		intra=False, 
		temperature=.5, 
		learning_rate=1e-4, 
		weight_decay=1e-6, 
		max_epochs=100, 
		hidden_dim=128, 
		top_k=10, 
		num_repeats=1, 
		dataset='ucm',
		technique='Repeat'
	):

		super(FullPipeline, self).__init__()
		self.batch_size = batch_size
		self.intra = intra
		self.temperature = temperature
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.hidden_dim = hidden_dim

		self.resnet_embedding_module = ImageEmbeddingModule()
		# Freeze weights
		# for param in self.resnet_embedding_module.parameters():
		# 	param.requires_grad = False


		self.bert_embedding_module = BERTSentenceEmbedding()
		# Freeze weights
		for param in self.bert_embedding_module.parameters():
			param.requires_grad = False

		self.projection_head = MyProjectionhead(512, 512, 128)
		
		self.criterion = SimCLRLoss(temperature)
		# self.criterion = NTXentLoss(temperature)
		self.max_epochs = max_epochs

		# To calculate all image embeddings at start of val and test epochs
		self.validation_embeddings = []
		self.test_embeddings = []

		self.validation_step_outputs = []
		self.test_step_outputs_1 = []
		self.test_step_outputs_5 = []
		self.test_step_outputs_10 = []
		self.test_step_outputs_20 = []

		self.test_step_ndcg_1 = []
		self.test_step_ndcg_5 = []
		self.test_step_ndcg_10 = []
		self.test_step_ndcg_20 = []

		self.test_bleu = []

		self.val_dataloader = val_dataloader
		self.test_dataloader = test_dataloader

		self.validation_labels = torch.Tensor()
		self.test_labels = torch.Tensor()

		self.top_k = top_k
		self.num_repeats = num_repeats

		self.dataset = dataset

		self.technique = technique

	def forward(self, batch):
		"""
		Forward pass through the model.

		Args:
			batch: Input batch with image, caption pairs.

		Returns:
			Tuple: Embeddings of the input pairs.
		"""

		image, caption = batch

		if self.intra:
			copy_img = image
			image = image[0], image[2]
			augmented_image = copy_img[1], copy_img[2]

			copy_caption = caption
			caption = caption[0], caption[2], caption[4]
			augmented_caption = copy_caption[1], copy_caption[3], copy_caption[4]

		image_embed = self.resnet_embedding_module(image)
		image_embed = image_embed.view(image_embed.size(0), -1)
		image_embed = self.projection_head(image_embed)

		caption_embed = self.bert_embedding_module(caption)
		caption_embed = self.projection_head(caption_embed)


		if self.intra:
			augmented_image_embed = self.resnet_embedding_module(augmented_image)
			augmented_image_embed = augmented_image_embed.view(augmented_image_embed.size(0), -1)
			augmented_image_embed = self.projection_head(augmented_image_embed)

			augmented_caption_embed = self.bert_embedding_module(augmented_caption)
			augmented_caption_embed = self.projection_head(augmented_caption_embed)

			return image_embed, augmented_image_embed, caption_embed, augmented_caption_embed
		
		return image_embed, caption_embed

	# only needs to be adjusted for mean feature, info, learned_fc, and learned_att
	def training_step(self, batch, batch_idx):
		"""
		Training step.

		Args:
			batch: Input batch for the training.
			batch_idx: Index of the current batch.

		Returns:
			torch.Tensor: NT-Xent loss inter modal (+optional intra modal)
		"""

		# NT-Xent loss between image and caption

		# if self.technique == 'RankAgg':

		# 	image, captions = batch

		# 	caption_emb_list = []

		# 	for idx, caption in enumerate(captions):

		# 		if self.intra:
		# 			_,_, caption_embed, _ = self((image, caption))

		# 		else:
		# 			_, caption_embed = self((image, caption))

		# 		caption_embed = F.normalize(caption_embed, dim=-1, p=2)

		# 		caption_emb_list.append(caption_embed)

		if self.technique == 'RankAgg':

			print('RankAgg technique')

			image, captions = batch

			caption_emb_list = torch.tensor([]).to('cuda:3')

			for idx, caption in enumerate(captions):

				if self.intra:
					image_embed, augmented_image_embed, caption_embed, _ = self((image, caption))

				else:
					image_embed, caption_embed = self((image, caption))

				caption_embed = F.normalize(caption_embed, dim=-1, p=2)

				torch.cat((caption_emb_list, caption_embed))

			print('caption embed list')
			print(caption_emb_list)

			caption_embed = torch.mean(caption_emb_list, axis=0)

			print('caption embed')
			print(caption_embed)

		else:
			if self.intra:
				image_embed, augmented_image_embed, caption_embed, augmented_caption_embed = self(batch)

			else:
				image_embed, caption_embed = self(batch)

		image_embed = F.normalize(image_embed, dim=-1, p=2)
		caption_embed = F.normalize(caption_embed, dim=-1, p=2)
		
		loss = self.criterion(image_embed, caption_embed)

		if self.intra:
			intra_image_loss = self.criterion(image_embed, augmented_image_embed)
			intra_caption_loss = self.criterion(caption_embed, augmented_caption_embed)

			loss = loss + intra_image_loss + intra_caption_loss

		self.log('train-loss', loss, prog_bar=True)
		return loss

	def calculate_embeddings_for_images(self, validation=True, true_label=False):
		"""
		Calculate and return embeddings for the entire set of images.

		Args:
			validation (bool): Whether to calculate embeddings for the validation or test set.

		Returns:
			List: Embeddings for the entire set of images.
		"""

		# TODO: This will also break for MEAN technique

		# Get the appropriate DataLoader
		dataloader = self.val_dataloader() if validation else self.test_dataloader()

		# List to store embeddings
		image_embeddings = []
		labels = []

		# Exists so we deduplicate the repeated images
		unique_embeddings = set()

		# Set to evaluation mode
		self.eval()

		# Offers speedup, don't calculate gradients
		with torch.no_grad():		
			# Get data batchwise
			for batch in dataloader:
				batch = to_cuda_recursive(batch, device=':3')

				# Retrieve image and caption
				image, caption = batch

				if self.intra:
					image = image[0], image[2]
					caption = caption[0], caption[2], caption[4]

				if self.technique in ['Mean', 'RankAgg', 'Info', 'Learned_FC', 'Learned_Att']:
					batch = batch[0], batch[1][0]
					image, caption = batch

				indeces = caption[2]

				# Get actual indices by dividing with repeats
				true_label_value = indeces // self.num_repeats

				# Forward pass to get image embeddings
				if self.intra:
					image_embed, _, _, _ = self(batch)

				else:
					image_embed, _ = self(batch)

				image_embed = F.normalize(image_embed, dim=-1, p=2)

				temp_embed = torch.tensor([]).to(image_embed.device)
				temp_labels = torch.tensor([]).to(image_embed.device)

				# Pass over all embeddings and labels
				for idx, embed in zip(true_label_value.tolist(), image_embed):
					# If this image was already seen skip it
					if idx in unique_embeddings:
						continue
					else:
						# Otherwise add the index to the dict
						unique_embeddings.add(idx)
						# Add embedding to list
						temp_embed = torch.cat([temp_embed, embed.unsqueeze(0).to(image_embed.device)], dim=0)

						# If we are not interested in the actual indeces (for testing)
						# Get class representations instead
						# UCM has 100 elements per class, NWPU has 700
						if not true_label:
							if self.dataset == 'ucm':
								idx = idx // 100
							elif self.dataset == 'nwpu':
								idx = idx // 700
						# Add the index (either true or class) to the list
						temp_labels = torch.cat([temp_labels, torch.tensor(idx).to(image_embed.device).unsqueeze(0)], dim=0)

				# Add all image embeddings and labels to return lists
				image_embeddings.append(temp_embed)
				labels.append(temp_labels)

		# Concatenate embeddings and labels
		image_embeddings = torch.cat(image_embeddings, dim=0)

		labels = torch.cat(labels, dim=0)

		#  Return embeddings and indeces (either true or class based)
		return image_embeddings, labels


	def shared_step(self, batch, validation=True):
		"""
		Shared step for both validation and test steps.

		Args:
			batch: Input batch for validation or test with image, caption pairs.

		Returns:
			np.array: mAP (mean Average Precision) values for input batch, based on embeddings.
		"""

		# To calculate bleu or not to calculate bleu
		ignore_bleu = True

		# Get image and caption
		image, caption = batch

		if self.intra:
			image = image[0], image[2]
			caption = caption[0], caption[2], caption[4]

		if self.technique in ['Mean', 'RankAgg', 'Info', 'Learned_FC', 'Learned_Att']:
			indeces = caption[0][2]

		else:
			indeces = caption[2]

		# From indeces to classes (either for ucm or nwpu)
		if self.dataset == 'ucm':
			labels_caption = indeces // (self.num_repeats * 100) # They need to be same if in same class
		elif self.dataset == 'nwpu':
			labels_caption = indeces // (self.num_repeats * 700)
		labels_images = self.validation_labels if validation else self.test_labels

		# Get indeces of relevant items for each embedding based on classes extracted above
		groundtruth = relevant_list(labels_caption, labels_images)

		# image_embed, augmented_image_embed, caption_embed, augmented_caption_embed

		# TODO: For mean feature and Rank aggregation 
		# caption batch now contains 5 captions each in a list pass through model to get list of embeddings each
		# need hyperparam if we're using multicaption
		# self.technique (Mean, RankAgg, Info, Learned_FC, Learned_Att)
		# if self.multicaption:
		# 	pass through model 5 times and save caption_embed as list

		# 1: MEAN FEATURE:
		# Pass through list of captions (5) get embeddings each and then mean --> store as caption embed

		if self.technique == 'Mean':

			print('Mean technique')

			image, captions = batch

			caption_emb_list = torch.tensor([]).to('cuda:3')

			for idx, caption in enumerate(captions):

				if self.intra:
					_,_, caption_embed, _ = self((image, caption))

				else:
					_, caption_embed = self((image, caption))

				caption_embed = F.normalize(caption_embed, dim=-1, p=2)

				torch.cat((caption_emb_list, caption_embed))

			caption_emb_list = torch.mean(caption_emb_list, axis=0)

			image_embeddings = self.validation_embeddings if validation else self.test_embeddings


		# 1.5: INFORMATIVENESS:
		# Pass through list of captions (5) calculate informativeness, get embeddings, mean them --> store as caption embed


		# 1.6: LEARNED WEIGHTS:
		# Pass through list of captions (5) get embeddings each, weighted by 
		# 	a) fc-layer
		# 	b) Attention mechanism
		# 	
		# 	mean with weights (softmax!) --> store as caption embed


		# 2: Rank Aggregation
		# Pass through list of captions (5) get embeddings each, store as list --> pass to calculate_mAP
		# 	--> calculates rank aggregated mAP

		if self.technique == 'RankAgg':

			image, captions = batch

			caption_emb_list = []

			for idx, caption in enumerate(captions):

				if self.intra:
					_,_, caption_embed, _ = self((image, caption))

				else:
					_, caption_embed = self((image, caption))

				caption_embed = F.normalize(caption_embed, dim=-1, p=2)

				caption_emb_list.append(caption_embed)

			image_embeddings = self.validation_embeddings if validation else self.test_embeddings

			(map_1,ndcg_1), (map_5,ndcg_5), (map_10,ndcg_10), (map_20,ndcg_20) = calculate_mAP(image_embeddings, caption_emb_list, groundtruth, top_k=1), calculate_mAP(image_embeddings, caption_emb_list, groundtruth, top_k=5), calculate_mAP(image_embeddings, caption_emb_list, groundtruth, top_k=10), calculate_mAP(image_embeddings, caption_emb_list, groundtruth, top_k=20)

		elif self.technique == 'Repeat':
			# Pass through model
			if self.intra:
				_, _, caption_embed, _ = self(batch)

			# image_embed, caption_embed
			else:
				_, caption_embed = self(batch)

			caption_embed = F.normalize(caption_embed, dim=-1, p=2)

		if not self.technique == 'RankAgg':
			# Get all image embeddings
			image_embeddings = self.validation_embeddings if validation else self.test_embeddings

			# mAP = calculate_mAP(image_embeddings, caption_embed, groundtruth, top_k=self.top_k) # multiple top k
			# Calculate mAP and Recall based on the groundtruth list constructed above
			(map_1,ndcg_1), (map_5,ndcg_5), (map_10,ndcg_10), (map_20,ndcg_20) = calculate_mAP(image_embeddings, caption_embed, groundtruth, top_k=1),  calculate_mAP(image_embeddings, caption_embed, groundtruth, top_k=5),  calculate_mAP(image_embeddings, caption_embed, groundtruth, top_k=10),  calculate_mAP(image_embeddings, caption_embed, groundtruth, top_k=20)

		if validation == False:

			if ignore_bleu:
				return (map_1,ndcg_1), (map_5,ndcg_5), (map_10,ndcg_10), (map_20,ndcg_20), -1

			bleu = calculate_bleu(image_embeddings, caption_embed, labels_images, caption)
			return (map_1,ndcg_1), (map_5,ndcg_5), (map_10,ndcg_10), (map_20,ndcg_20), bleu

		return (map_1,ndcg_1), (map_5,ndcg_5), (map_10,ndcg_10), (map_20,ndcg_20)


	def on_test_epoch_start(self):
		"""
		Called at the beginning of the test epoch.
		"""

		# Calculate and store embeddings for the entire test set.
		self.test_embeddings, self.test_labels = self.calculate_embeddings_for_images(validation=False)


	def test_step(self, batch, batch_idx):
		"""
		Test step.

		Args:
			batch: Input batch for testing. Image, caption pairs.
			batch_idx: Index of the current batch.
		"""

		(mAP_1,ndcg_1),(mAP_5,ndcg_5),(mAP_10,ndcg_10),(mAP_20,ndcg_20), bleu = self.shared_step(batch, validation=False)

		self.log('test_mAP_20',np.mean(mAP_20), batch_size=self.batch_size)

		self.test_step_outputs_1.append(mAP_1)
		self.test_step_outputs_5.append(mAP_5)
		self.test_step_outputs_10.append(mAP_10)
		self.test_step_outputs_20.append(mAP_20)

		self.test_step_ndcg_1.append(ndcg_1)
		self.test_step_ndcg_5.append(ndcg_5)
		self.test_step_ndcg_10.append(ndcg_10)
		self.test_step_ndcg_20.append(ndcg_20)

		# self.test_bleu.append(bleu)

	def on_test_epoch_end(self):
		"""
		Called at the end of the test epoch to calculate and log the average mAP.
		"""
		avg_mAP_1 = np.mean(np.concatenate(self.test_step_outputs_1))
		avg_mAP_5 = np.mean(np.concatenate(self.test_step_outputs_5))
		avg_mAP_10 = np.mean(np.concatenate(self.test_step_outputs_10))
		avg_mAP_20 = np.mean(np.concatenate(self.test_step_outputs_20))

		avg_ndcg_1 = np.mean(np.concatenate(self.test_step_ndcg_1))
		avg_ndcg_5 = np.mean(np.concatenate(self.test_step_ndcg_5))
		avg_ndcg_10 = np.mean(np.concatenate(self.test_step_ndcg_10))
		avg_ndcg_20 = np.mean(np.concatenate(self.test_step_ndcg_20))

		# avg_bleu = np.mean(np.concatenate(self.test_bleu))

		self.log('avg_test_mAP_1', avg_mAP_1, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
		self.log('avg_test_mAP_5', avg_mAP_5, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
		self.log('avg_test_mAP_10', avg_mAP_10, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
		self.log('avg_test_mAP_20', avg_mAP_20, batch_size=self.batch_size, prog_bar=True, sync_dist=True)

		self.log('avg_test_ndcg_1', avg_ndcg_1, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
		self.log('avg_test_ndcg_5', avg_ndcg_5, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
		self.log('avg_test_ndcg_10', avg_ndcg_10, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
		self.log('avg_test_ndcg_20', avg_ndcg_20, batch_size=self.batch_size, prog_bar=True, sync_dist=True)

		# self.log('avg_bleu', avg_bleu, batch_size=self.batch_size, prog_bar=True, sync_dist=True)


	def on_validation_epoch_start(self):
		"""
		Called at the beginning of the validation epoch.
		"""

		# Calculate and store embeddings for the entire validation set.
		self.validation_embeddings, self.validation_labels = self.calculate_embeddings_for_images(validation=True)


	def validation_step(self, batch, batch_idx):
		"""
		Validation step.

		Args:
			batch: Input batch for validation. Image, caption pairs.
			batch_idx: Index of the current batch. 
		"""

		_, _, _, (mAP,_) = self.shared_step(batch)
		self.log('validation mAP',np.mean(mAP), batch_size=self.batch_size)
		self.validation_step_outputs.append(mAP)

	def on_validation_epoch_end(self):
		"""
		Called at the end of the validation epoch to calculate and log the average mAP.
		"""
		avg_mAP = np.mean(np.concatenate(self.validation_step_outputs))
		self.log('avg_val_mAP', avg_mAP, batch_size=self.batch_size, prog_bar=True, sync_dist=True) # log all k

	def configure_optimizers(self):
		"""
		Configure the optimizer.

		Returns:
			toch.optim.Optimizer: Configured optimizer (AdamW)
		"""

		param_groups = define_param_groups(self, self.weight_decay, 'adam')

		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		# lr_scheduler = LinearWarmupCosineAnnealingLR(
		# 	optimizer, warmup_epochs=10, max_epochs=self.max_epochs, warmup_start_lr=self.learning_rate/10
		# )
		return optimizer
