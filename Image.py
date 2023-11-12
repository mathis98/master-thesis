import torch
import pytorch_lightning as pl
import torchvision
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

torchvision.disable_beta_transforms_warning()


data_dir = '../Datasets/UCM/imgs'
batch_size = 64
image_size = (224, 224)
simclr = True


# Embedding Only
data_module = ImageDataModule(data_dir, image_size, batch_size)
data_module.prepare_data()
data_module.setup()

image_embedding_model = ImageEmbeddingModule(image_size)


# SimCLR
augmentation_transform = v2.Compose([
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

simclr_data_module = SimCLRDataModule(data_dir, image_size, batch_size, augmentation_transform)
simclr_data_module.prepare_data()
simclr_data_module.setup(stage="fit")

simclr_module = SimCLRModule(image_size=image_size, batch_size=batch_size)

devices = find_usable_cuda_devices(1)
print(f'training on GPU {devices}')

trainer = pl.Trainer(accelerator='cuda', devices=devices)

if simclr:
	trainer.fit(simclr_module, simclr_data_module.train_dataloader())


	visualize_augmentations(simclr_data_module.train_dataset, 5)


	embeddings = simclr_module.embed_data(simclr_data_module.train_dataloader())	


else:
	summary(image_embedding_model)


	with torch.no_grad():
		predictions = trainer.predict(image_embedding_model, dataloaders=data_module.train_dataloader())

	embeddings = torch.vstack(predictions)
	embeddings = embeddings.view(embeddings.size(0), -1)

	
print('Shape: ', embeddings.shape)

pairs = closest_indices(embeddings)

for first, second in pairs:
	print(data_module.train_dataset.image_paths[first], '<-->', data_module.train_dataset.image_paths[second], ' (', first, ',', second, ')')