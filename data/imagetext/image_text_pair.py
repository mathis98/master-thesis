import pytorch_lightning as pl

class ImageTextPairDataModule(pl.LightningDataModule):
	def __init__(self, image_data_module, text_data_module, batch_size=64):
		super(ImageTextPairDataModule, self).__init__()
		self.image_data_module = image_data_module
		self.text_data_module = text_data_module
		self.batch_size = batch_size

	def prepare_data(self):
		pass

	def setup(self, stage=None):
		pass

	def train_dataloader(self):
		for _ in range(len(self.image_data_module.train_dataloader())):
			image_dataloader = list(self.image_data_module.train_dataloader())
			text_dataloader = list(self.text_data_module.train_dataloader())
			for image_batch, text_batch in zip(image_dataloader, text_dataloader):
				yield image_batch, text_batch

	def val_dataloader(self):
		for _ in range(len(self.image_data_module.val_dataloader())):
			image_dataloader = list(self.image_data_module.val_dataloader())
			text_dataloader = list(self.text_data_module.val_dataloader())
			for image_batch, text_batch in zip(image_dataloader, text_dataloader):
				yield image_batch, text_batch


	def test_dataloader(self):
		for _ in range(len(self.image_data_module.test_dataloader())):
			image_dataloader = list(self.image_data_module.test_dataloader())
			text_dataloader = list(self.text_data_module.test_dataloader())
			for image_batch, text_batch in zip(image_dataloader, text_dataloader):
				yield image_batch, text_batch