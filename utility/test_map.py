# logs/full_pipeline_full_val_test/version_193/checkpoints$ ls
# 'epoch=7-avg_val_mAP=0.36-validation mAP=0.38.ckpt'

import sys
sys.path.append('..')

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.accelerators import find_usable_cuda_devices

import torch
from model.full_pipeline import FullPipeline  # Import your model
from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataModule

from tsnecuda import TSNE
import matplotlib.pyplot as plt

# Argument parsing
from utility.argument_parser import parse_arguments
from utility.helpers import to_cuda_recursive

batch_size = 32


image_data_module = ImageDataModule('../../Datasets/UCM/imgs', (224,224), batch_size, 5)
image_data_module.prepare_data()
image_data_module.setup(stage='predict')


text_data_module = SentenceDataModule('prajjwal1/bert-small', batch_size, '../../Datasets/UCM/dataset.json')
text_data_module.prepare_data()
text_data_module.setup(stage='predict')

image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
image_text_pair_data_module.setup(stage='predict')


full_pipeline = FullPipeline.load_from_checkpoint(
	'../logs/full_pipeline_full_val_test/version_69/checkpoints/epoch=99-avg_val_mAP=0.34-validation mAP=0.47.ckpt',
	batch_size=batch_size,
	max_epochs=100,
	temperature=3.0,
	learning_rate=1e-4,
	weight_decay=1e-4,
	intra=False,
	top_k=20,
	val_dataloader=image_text_pair_data_module.val_dataloader,
	test_dataloader=image_text_pair_data_module.test_dataloader,
)


full_pipeline.eval()

devices = find_usable_cuda_devices(1)
print(f'training on GPU {devices}')

trainer = pl.Trainer(accelerator='cuda', devices=[3], max_epochs=100)

trainer.test(full_pipeline, dataloaders=image_text_pair_data_module.test_dataloader())
