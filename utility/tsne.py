import sys
sys.path.append('..')

from lightning.pytorch  import seed_everything

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

seed_everything(42, workers=True)


image_data_module = ImageDataModule('../../Datasets/UCM/imgs', (224,224), 512, 5)
image_data_module.prepare_data()
image_data_module.setup(stage='predict')


text_data_module = SentenceDataModule('prajjwal1/bert-small', 512, '../../Datasets/UCM/dataset.json')
text_data_module.prepare_data()
text_data_module.setup(stage='predict')

image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, 512)
image_text_pair_data_module.setup(stage='predict')

dataloader = image_text_pair_data_module.dataloader()

embeddings_list = []
labels_list = []

with torch.no_grad():
	for batch in dataloader:
		# NO SIMCLR: ((image, image_path), (inputs, sentence, index))
		batch = to_cuda_recursive(batch)
		embeddings = model.forward(batch)

		combined_embeddings = torch.cat(embeddings, dim=0)
		labels = batch[1][2]

		labels = labels // 500

		embeddings_list.append(combined_embeddings)
		labels_list.append(labels)

combined_embeddings = torch.cat(embeddings_list, dim=0)
labels = torch.cat(labels_list, dim=0)

tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(combined_embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')

plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE Visualization of Image and Caption Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.savefig('tsne.png')

