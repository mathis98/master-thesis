import sys
sys.path.append('..')

import pytorch_lightning as pl
import torch
import numpy as np

# Embedding for text
from model.text_embedding import BERTSentenceEmbedding
from model.simclr_text_model import SimCLRModule as SimCLRTextModule

# Embedding for image
from model.image_embedding import ImageEmbeddingModule
from model.simclr_image_model import SimCLRModule as SimCLRImageModule

# SimCLR loss
from loss.contrastive_loss import SimCLRLoss

# for mAP calculation
from utility.helpers import relevant_list, calculate_mAP


class FullPipeline(pl.LightningModule):
	def __init__(self, batch_size=64, simclr=False, temperature=.07, learning_rate=1e-4):
		super(FullPipeline, self).__init__()
		self.batch_size = batch_size
		self.simclr = simclr
		self.temperature = temperature
		self.learning_rate = learning_rate

		if self.simclr:
			self.image_simclr_model = SimCLRImageModule()
			self.text_simclr_model = SimCLRTextModule()
			self.automatic_optimization = False

		else:
			self.bert_embedding_module = BERTSentenceEmbedding()
			self.resnet_embedding_module = ImageEmbeddingModule()
		
		self.criterion = SimCLRLoss(temperature)

		self.validation_step_outputs = []
		self.test_step_outputs = []

	def forward(self, image, caption):

		if self.simclr == True:
			image_embed = self.image_simclr_model(image[0], image[1])
			text_embed = self.text_simclr_model(caption[0], caption[1])

		else:
			image_embed = self.resnet_embedding_module(image)
			text_embed = self.bert_embedding_module(caption)
		
		return image_embed, text_embed

	def configure_optimizers(self):

		if self.simclr == True:
			image_optimizer = self.image_simclr_model.configure_optimizers()
			text_optimizer = self.text_simclr_model.configure_optimizers()
			return [image_optimizer, text_optimizer]

		else:
			optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
			return optimizer

	def training_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		if self.simclr == True:

			orig_img, aug_img, img_path, source_img = image
			inputs, inputs_aug, sentence, sentence_aug, idx = caption

			image = (orig_img, img_path)
			image_aug = (aug_img, img_path)
			source_image = (source_img, img_path)

			caption = (inputs, sentence, idx)
			caption_aug = (inputs_aug, sentence_aug, idx)

			image_embed, caption_embed = self((source_img, source_img),(inputs, inputs))
			image_embed = image_embed[0]
			caption_embed = caption_embed[0]

		else:

			image_embed, caption_embed = self(image, caption)
		
		image_embed = torch.squeeze(image_embed)
		
		loss = self.criterion(image_embed, caption_embed)

		if self.simclr  == True:
			image_loss = self.image_simclr_model.training_step((orig_img, aug_img), batch_idx)
			text_loss = self.text_simclr_model.training_step(batch[1], batch_idx)

			loss = loss + image_loss + text_loss

			self.manual_backward(loss)
			self.image_simclr_model.optimizer.step()
			self.text_simclr_model.optimizer.step()

		self.log('train-loss', loss, batch_size=self.batch_size)
		return loss

	def test_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		if self.simclr:

			orig_img, aug_img, img_path, source_img = image
			inputs, inputs_aug, sentence, sentence_aug, idx = caption

			image = (orig_img, img_path)
			image_aug = (aug_img, img_path)
			source_image = (source_img, img_path)

			caption = (inputs, sentence, idx)
			caption_aug = (inputs_aug, sentence_aug, idx)

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		if self.simclr == True:
			image_embed, caption_embed = self((source_img, source_img),(inputs, inputs))
			image_embed = image_embed[0]
			caption_embed = caption_embed[0]

		else:
			image_embed, caption_embed = self(image, caption)
		
		image_embed = torch.squeeze(image_embed)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('train mAP:',mAP)
		self.test_step_outputs.append(mAP)
		return mAP

	def on_test_epoch_end(self):
		avg_mAP = np.mean(self.test_step_outputs)
		self.log('avg_test_mAP: ', avg_mAP, batch_size=self.batch_size)
		print('avg_test_mAP: ', avg_mAP)


	def validation_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		if self.simclr:

			orig_img, aug_img, img_path, source_img = image
			inputs, inputs_aug, sentence, sentence_aug, idx = caption

			image = (orig_img, img_path)
			image_aug = (aug_img, img_path)
			source_image = (source_img, img_path)

			caption = (inputs, sentence, idx)
			caption_aug = (inputs_aug, sentence_aug, idx)

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		if self.simclr:
			image_embed, caption_embed = self((source_img, source_img),(inputs, inputs))
			image_embed = image_embed[0]
			caption_embed = caption_embed[0]

		else:
			image_embed, caption_embed = self(image, caption)
	
		image_embed = torch.squeeze(image_embed)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('validation mAP:',mAP)
		self.validation_step_outputs.append(mAP)
		return mAP

	def on_validation_epoch_end(self):
		avg_mAP = np.mean(self.validation_step_outputs)
		self.log('avg_val_mAP: ', avg_mAP, batch_size=self.batch_size)
		print('avg_val_mAP: ', avg_mAP)