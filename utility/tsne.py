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


# Create an instance of your FullPipeline model
model = FullPipeline.load_from_checkpoint('../logs/full_pipeline_full_val_test/version_35/checkpoints/epoch=99-avg_val_mAP=0.28-validation mAP=0.26.ckpt')

# Ensure the model is in evaluation mode
model.eval()


image_data_module = ImageDataModule('../../Datasets/UCM/imgs', (224,224), 512, 5)
image_data_module.prepare_data()
image_data_module.setup(stage='predict')


text_data_module = SentenceDataModule('prajjwal1/bert-small', 512, '../../Datasets/UCM/dataset.json')
text_data_module.prepare_data()
text_data_module.setup(stage='predict')

image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, 512)
image_text_pair_data_module.setup(stage='predict')

dataloader = image_text_pair_data_module.dataloader()

devices = find_usable_cuda_devices(1)
print(f'training on GPU {devices}')

trainer = pl.Trainer(accelerator='cuda', devices=devices, max_epochs=100)

with torch.no_grad():
		predictions = trainer.predict(model, dataloader)

print(predictions[0])

print(predictions[0][0])
print(predictions[0][1])

print(len(predictions[0]))

print(len(predictions))
print(predictions[:2])

image_embeddings = torch.vstack(predictions[0])
image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)

caption_embeddings = torch.vstack(predictions[1])
caption_embeddings = caption_embeddings.view(caption_embeddings.size(0), -1)


print(len(image_embeddings))

print(len(caption_embeddings))

labels_simple = np.repeat(range(21), 500)
labels = np.repeat(labels_simple, 2)

all_embeddings = torch.cat([image_embeddings, caption_embeddings], dim=0).cpu().numpy()

print(len(all_embeddings))

print(all_embeddings[:2])

tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')

plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE Visualization of Image and Caption Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.savefig('tsne.png')

