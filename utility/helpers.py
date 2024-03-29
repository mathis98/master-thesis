from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_normalized_dcg
from torchmetrics.functional.text import bleu_score
from transformers.tokenization_utils_base import BatchEncoding
import json
import torch.nn.functional as F


def closest_indices(embeddings):
	"""
	Find the indices of the closes embedding pairs based on cosine similarity.

	Args:
		embeddings (numpy.ndarray): The embeddings for which to find the closest indices.

	Returns:
		list: List of tuples representing the indices of the closest pairs.
	"""

	cosine_similarities = cosine_similarity(embeddings, embeddings)
	np.fill_diagonal(cosine_similarities, -np.inf)

	top_indices = np.unravel_index(np.argpartition(cosine_similarities, -5, axis=None)[-5:], cosine_similarities.shape)
	pairs = list(zip(top_indices[0], top_indices[1]))

	return pairs


def visualize_augmentations(data, number, mean, std):
	"""
	Visualize augmented image pairs.

	Args:
		data (list): List of image pairs to visualize.
		number (int): Number of pairs to visualize.
		mean (list): Mean values for denormalization.
		std (list): Standard deviation values for denormalization.
	"""

	fig, axes = plt.subplots(number, 2, figsize=(8,10))

	column_titles = ["First", "Second"]

	for row in range(number):

		images = data[row]
		original = images[0]
		augmented = images[1]


		unnormalize = v2.Normalize(
			mean=[-m / s for m, s in zip(mean,std)],
			std=[1 / s for s in std]
		)
		original = unnormalize(original).numpy().transpose(1,2,0)
		augmented = unnormalize(augmented).numpy().transpose(1,2,0)

		axes[row, 0].imshow(original)
		axes[row, 0].set_title(column_titles[0])
		axes[row, 0].axis('off')

		axes[row, 1].imshow(augmented)
		axes[row, 1].set_title(column_titles[1])
		axes[row, 1].axis('off')

	plt.tight_layout()
	plt.show()


def visualize_text_augmentations(data, number):
	"""
	Print pairs of original and augmented captions.

	Args:
		data (list): List of caption pairs to visualize.
		number (int): Number of pairs to visualize.
	"""

	for idx in range(number):

		sentences = data[idx]
		original = sentences[2]
		augmented = sentences[3]

		print(original, '->', augmented)

def relevant_list(labels_caption, labels_images):
	"""
	Create a list of relevant indices for each label (index//#elems per class).

	Args:
		labels (torch.Tensor): indices transformed to class labels.
	
	Returns:
		list: List of relevant indices for each label.
	"""

	relevant_list = []

	for label_tensor in labels_caption:
		label_value = label_tensor.item() if label_tensor.numel() == 1 else label_tensor

		relevants = (labels_images == label_value).to(torch.bool)
		relevant_list.append(relevants)

	return relevant_list


def calculate_mAP(image_embeddings, caption_embeddings, ground_truth_labels, top_k=10):
	"""
	Calculate mean Average Precision (mAP) values.

	Args:
		image_embeddings (torch.Tensor): Image embeddings.
		caption_embeddings (torch.Tensor): Caption embeddings.
		ground_truth_labels (torch.Tensor): list of lists of relevant labels.
		top_k (int): Top-k retrieval (map@k).

	Returns:
		list: List of mAP values for each input caption.
	"""

	# TODO: if technique is Rank Aggregation (multicaption=true) 
	# 		--> for each caption embedding we have a list of 5 embeddings each
	#       calculate image scores for each and mean them
	#       THEN calculate mAP and NDCG!

	mAP_values = []

	ndcg_values = []

	if(isinstance(caption_embeddings, list)):

		# captions1: [1,2,3,4]
		# captions2: [1,2,3,4]

		# go through each image (now actually multiple)
		for idx in range(len(caption_embeddings[0])):
			image_scores_list = []

			# go through each caption for this image (captions1[0], captions2[1],...)
			for idx2 in range(len(caption_embeddings)):

				# get embedding for this specific caption
				caption_embedding = caption_embeddings[idx2][idx]

				# calculate cosine similarity with image embeddings
				image_scores = torch.matmul(image_embeddings, caption_embedding)

				# add to list
				image_scores_list.append(image_scores.cpu().numpy())

			# take mean for rank aggregation
			image_scores = torch.tensor(np.mean(image_scores_list, axis=0)).to('cuda:3')

			# calculate mAP and recall based on this mean
			relevant_labels = ground_truth_labels[idx]

			mAP = retrieval_average_precision(
				image_scores, 
				relevant_labels, 
				top_k=top_k
			)

			recall = retrieval_normalized_dcg(
				image_scores,
				relevant_labels,
				top_k=top_k,
			)

			# add mAP and ndcg to list to have this for every caption
			mAP_values.append(mAP.cpu().numpy())
			ndcg_values.append(recall.cpu().numpy())

		# return map and ndcg lists
		return mAP_values, ndcg_values

	# captions: [1,2,3,4]
	for i in range(caption_embeddings.shape[0]):

		caption_embedding = caption_embeddings[i]
		
		image_scores = torch.matmul(image_embeddings, caption_embedding)

		relevant_labels = ground_truth_labels[i]

		mAP = retrieval_average_precision(
			image_scores, 
			relevant_labels, 
			top_k=top_k
		)

		recall = retrieval_normalized_dcg(
			image_scores,
			relevant_labels,
			top_k=top_k,
		)

		mAP_values.append(mAP.cpu().numpy())
		ndcg_values.append(recall.cpu().numpy())

	return mAP_values, ndcg_values

# Change this to compute pairwise bleus between retrieved and groundtruth captions
# get mean and max bleu scores
def calculate_bleu(image_embeddings, caption_embeddings, image_labels, captions):
	bleu_values = []

	for i in range(caption_embeddings.shape[0]):

		caption_embedding = caption_embeddings[i]
		caption = captions[1][i] # Herein lies the problem!

		image_scores = torch.matmul(image_embeddings, caption_embedding)

		image_label = image_labels[torch.argsort(image_scores, descending=True)[:1]]

		bleu = bleu_score([caption], [get_ground_truth_captions(image_label)], n_gram=2)

		# print(bleu) # It works but is reallly slow, also how to proceed with multiple captions?

		bleu_values.append(bleu)

	return bleu_values


def calculate_uniqueness(captions):
	uniqueness_scores = []

	for idx,caption in enumerate(captions):
		reference = captions[:idx] + captions[idx+1:]

		bleu = bleu_score([caption], [reference], n_gram=2)
		uniqueness = 1 - bleu

		uniqueness_scores.append(uniqueness)

	uniqueness_scores = torch.tensor(uniqueness_scores)
	uniqueness_scores = F.softmax(uniqueness_scores, dim=0)

	return uniqueness_scores


def define_param_groups(model, weight_decay, optimizer_name):
	"""
	Define paramter groups for optimization. Remove weight_decay from batch normalization layers.

	Args:
		model (torch.nn.Module): Pytorch model to define param groups on.
		weight_decay (float): Weight decay to apply to non-bn layers.
		optimizer_name (str): Name of optimizer.

	Returns:
		list: List of dictionaries containing bn (no weight-decay) and non-bn (weight-decay) layers.
	"""

	return[
		{
			'params': [p for name, p in model.named_parameters() if not 'bn' in name],
			'weight_decay': weight_decay,
			'layer_adaptation': True,
		},
		{
			'params': [p for name, p in model.named_parameters() if 'bn' in name],
			'weight_decay': 0.,
			'layer_adaptation': False,
		},
	]


def to_cuda_recursive(obj, device=''):
	if isinstance(obj, torch.Tensor):
		# Move the tensor to the CUDA device
		return obj.to('cuda:3')
	elif isinstance(obj, list):
		# Recursively move each element of the list to the CUDA device
		return [to_cuda_recursive(item) for item in obj]
	elif isinstance(obj, tuple):
		# Recursively move each element of the tuple to the CUDA device
		return tuple(to_cuda_recursive(item) for item in obj)
	elif isinstance(obj, dict):
		# Recursively move each value of the dictionary to the CUDA device
		return {key: to_cuda_recursive(value) for key, value in obj.items()}
	elif isinstance(obj, BatchEncoding):
		# Handle BatchEncoding separately
		obj.data = to_cuda_recursive(obj.data)
		return obj
	elif isinstance(obj, np.ndarray):
		# Handle numpy arrays
		return torch.from_numpy(obj).to('cuda:3')
	else:
		return obj  # Return unchanged if not a tensor, list, tuple, or dict


def get_ground_truth_captions(idx,dataset='nwpu',num_repeats=1):


	idx = int(idx.cpu().numpy())

	if dataset == 'ucm':
		text_path = '../Datasets/UCM/dataset.json'

	elif dataset == 'nwpu':
		text_path = '../Datasets/NWPU-Captions-main/dataset_nwpu.json'

	with open(text_path, 'r') as json_file:
			data = json.load(json_file)

	if dataset == 'nwpu':
		# concatenate raw, raw_1, raw_2, raw_3, and raw_4
		categories = sorted([category for category in data])

		category = categories[idx // (700 * num_repeats)]
		index = idx % 700

		item = data[category][index]

		sentences = ([item['raw']] + [item[f'raw_{i}'] for i in range(1, 5)])

	# UCM dataset
	else:
		# concatentate 'raw' for 'sentences'[1-4]

		item = data['images'][idx]
		sentences = [item['sentences'][i]['raw'] for i in range(5)]

	return sentences


