import lightning.pytorch as pl
import torch
from model.full_pipeline import FullPipeline
from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataModule
from data.image.simclr_data_module import SimCLRDataModule as SimCLRImageDataModule
from data.text.simclr_data_module import SimCLRDataModule as SimCLRTextDataModule
from torchvision.transforms import v2
from transformers import AutoTokenizer
import random
import os

batch_size = 512

intra = False

augmentation_transform = v2.Compose([
		v2.Resize((224,224)),
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')


if intra == True:
	image_data_module = SimCLRImageDataModule('../Datasets/UCM/imgs', (224,224), batch_size, augmentation_transform)
	image_data_module.prepare_data()
	image_data_module.setup(stage="predict")

	text_data_module = SimCLRTextDataModule(batch_size, '../Datasets/UCM/dataset.json', tokenizer)
	text_data_module.prepare_data()
	text_data_module.setup()

elif intra == False:
	image_data_module = ImageDataModule('../Datasets/UCM/imgs', (224,224), batch_size, 5)
	image_data_module.prepare_data()
	image_data_module.setup(stage='predict')


	text_data_module = SentenceDataModule('prajjwal1/bert-small', batch_size, '../Datasets/UCM/dataset.json')
	text_data_module.prepare_data()
	text_data_module.setup(stage='predict')

image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
image_text_pair_data_module.setup(stage='predict')

version = input('Version number to load: ')


if version == '':
	print('Loading untrained model')

	full_pipeline = FullPipeline(
		batch_size=batch_size, 
		max_epochs=1, 
		temperature=3.0, 
		learning_rate=1e-4, 
		weight_decay=1e-4, 
		intra=intra,
		top_k=20,
		val_dataloader = image_text_pair_data_module.val_dataloader,
		test_dataloader = image_text_pair_data_module.test_dataloader,
	)

else:
	name = os.listdir(f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints')[0]
	checkpoint = f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints/{name}'

	print(f'Loading from {checkpoint}')

	full_pipeline = FullPipeline.load_from_checkpoint(
		checkpoint,
		batch_size=batch_size, 
		max_epochs=1, 
		temperature=3.0, 
		learning_rate=1e-4, 
		weight_decay=1e-4, 
		intra=intra,
		top_k=20,
		val_dataloader = image_text_pair_data_module.val_dataloader,
		test_dataloader = image_text_pair_data_module.test_dataloader,
	)

device = 'cuda:3'

full_pipeline.resnet_embedding_module.to(device)
full_pipeline.bert_embedding_module.to(device)
full_pipeline.projection_head.to(device)

image_text_pair_data_module.device = device


full_pipeline.eval()

image_embeddings, labels = full_pipeline.calculate_embeddings_for_images(validation=False, true_label=True)

random_sample = random.sample(list(text_data_module.test_dataset), 5)

print('5 Random samples:')
for element in random_sample:
	print(f'Sentence: {element[1]} (Index: {element[2] // 5 + 1})')

while True:

	query = input('Enter query caption (Ctrl + C to exit): ')
	if not query:
		break

	caption = tokenizer(query, return_tensors='pt').to(device)
	new_caption = [caption]
	new_caption_embedding = full_pipeline.bert_embedding_module(new_caption)
	new_caption_projection = full_pipeline.projection_head(new_caption_embedding)

	similarity_scores = torch.nn.functional.cosine_similarity(image_embeddings, new_caption_projection)

	top_k = 20
	sorted_indices = torch.argsort(similarity_scores, descending=True)[:top_k]

	print('20 closest images:')
	for idx in sorted_indices:
		print(f'Image index: {int(labels[idx].item())}, Similarity: {similarity_scores[idx].item()}')
