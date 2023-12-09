# logs/full_pipeline_full_val_test/version_193/checkpoints$ ls
# 'epoch=7-avg_val_mAP=0.36-validation mAP=0.38.ckpt'

import sys
sys.path.append('..')

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from transformers import AutoTokenizer

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.accelerators import find_usable_cuda_devices

import torch
from model.full_pipeline import FullPipeline  # Import your model
from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataModule

from data.image.simclr_data_module import SimCLRDataModule as SimCLRImageDataModule
from data.text.simclr_data_module import SimCLRDataModule as SimCLRTextDataModule

from tsnecuda import TSNE
import matplotlib.pyplot as plt

# Argument parsing
from utility.argument_parser import parse_arguments
from utility.helpers import to_cuda_recursive

batch_size = 256

intra = False

augmentation_transform = v2.Compose([
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')


if intra == True:
	image_data_module = SimCLRImageDataModule('../../Datasets/UCM/imgs', (224,224), batch_size, augmentation_transform)
	image_data_module.prepare_data()
	image_data_module.setup(stage="predict")

	text_data_module = SimCLRTextDataModule(batch_size, '../../Datasets/UCM/dataset.json', tokenizer)
	text_data_module.prepare_data()
	text_data_module.setup()

elif intra == False:
	image_data_module = ImageDataModule('../../Datasets/UCM/imgs', (224,224), batch_size, 5)
	image_data_module.prepare_data()
	image_data_module.setup(stage='predict')


	text_data_module = SentenceDataModule('prajjwal1/bert-small', batch_size, '../../Datasets/UCM/dataset.json')
	text_data_module.prepare_data()
	text_data_module.setup(stage='predict')

image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
image_text_pair_data_module.setup(stage='predict')


full_pipeline = FullPipeline(
	batch_size=batch_size, 
	max_epochs=100, 
	temperature=3.0, 
	learning_rate=1e-4, 
	weight_decay=1e-4, 
	intra=intra,
	top_k=20,
	val_dataloader = image_text_pair_data_module.val_dataloader,
	test_dataloader = image_text_pair_data_module.test_dataloader,
)


full_pipeline.eval()

# devices = find_usable_cuda_devices(1)
# print(f'training on GPU {devices}')

trainer = pl.Trainer(accelerator='cuda', devices=[3], max_epochs=100)

trainer.test(full_pipeline, dataloaders=image_text_pair_data_module.test_dataloader())
