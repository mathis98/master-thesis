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
import yaml
import numpy as np
import inquirer

batch_size = 512
num_repeats = 5

intra = False

augmentation_transform = v2.Compose([
		v2.Resize((224,224)),
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

questions = [
	inquirer.List(
		'technique',
		message= 'Select a technique to evaluate on:',
		choices=[
			'Repeat',
			'Random',
			'Concat',
			'Mean', 
			'RankAgg', 
			'Info', 
			'Learned_FC', 
			'Learned_Att',
		],
	),
]

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

version = input('Version number to load: ')


if version == '':
	print('Loading untrained model')

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

		text_data_module = SentenceDataModule('prajjwal1/bert-small', batch_size, '../Datasets/UCM/dataset.json', 5)
		text_data_module.prepare_data()
		text_data_module.setup(stage='predict')

	hparams = {'num_repeats':5}

	image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
	image_text_pair_data_module.setup(stage='predict')

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
		dataset='ucm',
		num_repeats=5,
	)

else:
	name = os.listdir(f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints')[0]
	checkpoint = f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints/{name}'

	print(f'Loading from {checkpoint}')

	with open(f'./logs/full_pipeline_full_val_test/version_{version}/hparams.yaml') as file:
		hparams = yaml.safe_load(file)

	retrieval_technique = inquirer.prompt(questions)['technique']

	if not 'technique' in hparams:
		hparams['technique'] = 'Repeat'

	if not 'dataset' in hparams:
		hparams['dataset'] = 'ucm'

	print(hparams)

	image_data_module = ImageDataModule(hparams['img_path'], tuple(hparams['image_size']), hparams['batch_size'], num_repeats=hparams['num_repeats'], technique=hparams['technique'])
	image_data_module.prepare_data()
	image_data_module.setup(stage='predict')

	text_data_module = SentenceDataModule(hparams['model_name'], hparams['batch_size'], hparams['text_path'], num_repeats=hparams['num_repeats'], technique=hparams['technique'])
	text_data_module.prepare_data()
	text_data_module.setup(stage='predict')

	image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, hparams['batch_size'])
	image_text_pair_data_module.setup(stage='predict')

	full_pipeline = FullPipeline.load_from_checkpoint(
		checkpoint,
		batch_size=hparams['batch_size'], 
		max_epochs=hparams['max_epochs'], 
		temperature=hparams['temperature'], 
		learning_rate=hparams['learning_rate'], 
		weight_decay=hparams['weight_decay'], 
		intra=hparams['intra'],
		top_k=hparams['top_k'],
		val_dataloader = image_text_pair_data_module.val_dataloader,
		test_dataloader = image_text_pair_data_module.test_dataloader,
		dataset=hparams['dataset'],
		num_repeats=hparams['num_repeats'],
	)

device = 'cuda:3'

full_pipeline.resnet_embedding_module.to(device)
full_pipeline.bert_embedding_module.to(device)
full_pipeline.projection_head.to(device)

image_text_pair_data_module.device = device


full_pipeline.eval()

image_embeddings, labels = full_pipeline.calculate_embeddings_for_images(validation=False, true_label=True)


if hparams['dataset'] == 'nwpu':
	categories = []
	cats = sorted(os.listdir(hparams['img_path']))
	for category in cats:
		if os.path.isdir(os.path.join('../Datasets/NWPU-Captions-main/NWPU-RESISC45', category)):
			categories.append(category)

random_sample = random.sample(list(text_data_module.test_dataset), 5)

print('5 Random samples:')
for element in random_sample:

	name = ''
	index = element[2] // hparams["num_repeats"]

	if hparams['dataset'] == 'nwpu':
		category_index = index // 700
		elem_index = index % 700

		name = f' ↔ {categories[category_index]}_{elem_index}'

	print(f'Sentence: {element[1]} (Index: {index}{name})')

# 'Repeat',
# 'Random',
# 'Concat',
# 'Mean', 
# 'RankAgg', 
# 'Info', 
# 'Learned_FC', 
# 'Learned_Att',

def calc_sim_scores_for_single(query):
	caption = tokenizer(query, return_tensors='pt').to(device)
	new_caption = [caption]
	new_caption_embedding = full_pipeline.bert_embedding_module(new_caption)
	new_caption_projection = full_pipeline.projection_head(new_caption_embedding)

	# Calculate pairwise cosine similarity between all image embeddings and the projected query
	return torch.nn.functional.cosine_similarity(image_embeddings, new_caption_projection)

def shorten_query(queries):
	return f'{queries[0]} [...]'

while True:

	# Single query only for Repeat technique
	if retrieval_technique == 'Repeat':
		query = input('Enter query caption (Ctrl + C to exit): ')

		# Break on empty query
		if not query:
			break

		# TODO: if hparams[technique] = RankAgg, Mean, Info, Learned_FC, Learned_Att
		# enable multiple queries to be provided
		# For Mean: get embeddings for all captions and mean
		# For Info: get embeddings for all captions and mean weighted by informativeness
		# For Learned_FC: get embeddings for all captions and mean weighted by learned FC layer
		# For Learned_Att: get embeddings for all captions and mean weighted by learned Attention layer
		# For RankAgg: get embeddings for all captions get similarity_scores for these embeddings, mean them, get indices

		similarity_scores = calc_sim_scores_for_single(query)

	# Allow entering multiple queries otherwise
	else:

		if retrieval_technique in ['Mean', 'Info', 'Learned_FC', 'Learned_Att']:
			print('These techniques are not yet implemented!')

			quit()

		queries = []
		while True:
			query = input('Enter next caption (Enter to evaluate, stops if no captions entered): ')
			if not query:
				break

			queries.append(query)

		if len(queries) == 0:
			break

		print(f'All queries you have entered: {queries}')

		if retrieval_technique == 'Random':

			query = queries[3]

			similarity_scores = calc_sim_scores_for_single(query)

		elif retrieval_technique == 'Concat':
			sentences = []
			query = ' '.join(queries)

			similarity_scores = calc_sim_scores_for_single(query)
			query = shorten_query(queries)


		# For Rank Aggregation mean the image positions
		elif retrieval_technique == 'RankAgg':
			image_scores_list = []

			for caption in queries:

				image_scores = calc_sim_scores_for_single(caption)
				image_scores_list.append(image_scores.detach().cpu().numpy())

			# take mean for rank aggregation
			similarity_scores = torch.tensor(np.mean(image_scores_list, axis=0)).to('cuda:3')
			query = shorten_query(queries)

	# Only retrieve the top 20 indices of biggest cosine similarity
	top_k = 20
	sorted_indices = torch.argsort(similarity_scores, descending=True)[:top_k]

	idxs = []

	print('20 closest images:')
	for idx in sorted_indices:

		name = ''

		index = int(labels[idx].item())
		idxs.append(index)

		if hparams['dataset'] == 'nwpu':
			category_index = index // 700
			elem_index = index % 700

			name = f' ↔ {categories[category_index]}_{elem_index}'

		print(f'Image index: {index}{name}, Similarity: {similarity_scores[idx].item()}')

	# Format output for direct usage in visualize.py script which visualizes retrieved images
	output = {
		'dataset': hparams['dataset'],
		'query': query,
		'idxs': idxs
	}

	print(output)
