import torch
import pytorch_lightning as pl
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchinfo import summary
import matplotlib.pyplot as plt
from lightning.pytorch.accelerators import find_usable_cuda_devices

# Data Modules
from data.image.data_module import ImageDataModule
from data.image.simclr_data_module import SimCLRDataModule

#  Embedding Modules
from model.image_embedding import ImageEmbeddingModule
from model.simclr_image_model import SimCLRModule

# Helper functions
from utility.helpers import closest_indices, visualize_augmentations


data_dir = '../Datasets/UCM/imgs'
batch_size = 256
image_size = (224, 224)
max_epochs = 500
hidden_dim = 128
lr = 5e-4
temperature = .07
weight_decay = 1e-4
simclr = True


# Embedding Only
data_module = ImageDataModule(data_dir, image_size, batch_size)
data_module.prepare_data()
data_module.setup()

image_embedding_model = ImageEmbeddingModule(image_size)


# SimCLR
augmentation_transform = v2.Compose([
		v2.RandomResizedCrop(size=image_size),
		# v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.RandomHorizontalFlip(p=.5),
		v2.RandomVerticalFlip(p=.5),
		v2.RandomRotation(10),
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
		v2.Normalize(mean=[0.4845, 0.4903, 0.4508],std=[0.2135, 0.1970, 0.1911]),
])

simclr_data_module = SimCLRDataModule(data_dir, image_size, batch_size, augmentation_transform, num_repeats=5)
simclr_data_module.prepare_data()
simclr_data_module.setup()

simclr_module = SimCLRModule(
	image_size=image_size, 
	max_epochs=max_epochs,
	temperature=temperature,
	learning_rate=lr,
	weight_decay=weight_decay,
	hidden_dim=hidden_dim,
)

devices = find_usable_cuda_devices(1)
print(f'training on GPU {devices}')

trainer = pl.Trainer(accelerator='cuda', devices=devices, max_epochs=max_epochs)

if simclr:

	summary(simclr_module)

	trainer.fit(simclr_module, simclr_data_module.train_dataloader())


	# visualize_augmentations(simclr_data_module.train_dataset, 5, mean=[0.4845, 0.4903, 0.4508],std=[0.2135, 0.1970, 0.1911])

	simclr_data_module_single = SimCLRDataModule(data_dir, image_size, batch_size, augmentation_transform, num_repeats=1)
	simclr_data_module_single.prepare_data()
	simclr_data_module_single.setup()

	with torch.no_grad():
		predictions = trainer.predict(simclr_module, dataloaders=simclr_data_module_single.train_dataloader())
		predictions = [elem[0] for elem in predictions]

else:
	summary(image_embedding_model)

	with torch.no_grad():
		predictions = trainer.predict(image_embedding_model, dataloaders=data_module.dataloader())

embeddings = torch.vstack(predictions)
embeddings = embeddings.view(embeddings.size(0), -1)

	
print('Shape: ', embeddings.shape)

pairs = closest_indices(embeddings)

if simclr:
	data_module = simclr_data_module_single

for first, second in pairs:
	print(data_module.train_dataset.image_paths[first], '<-->', data_module.train_dataset.image_paths[second])