from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2
from torchmetrics.retrieval import RetrievalMAP
from transformers.tokenization_utils_base import BatchEncoding

torchvision.disable_beta_transforms_warning()

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

def relevant_list(labels):
	"""
	Create a list of relevant indices for each label (index//#elems per class).

	Args:
		labels (torch.Tensor): indices transformed to class labels.
	
	Returns:
		list: List of relevant indices for each label.
	"""

	relevant_list = []

	for label in labels:
		relevants = torch.where(labels == label, True, False)
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

	mAP_values = []

	for i in range(caption_embeddings.shape[0]):

		caption_embedding = caption_embeddings[i]
		
		image_scores = torch.matmul(image_embeddings, caption_embedding)

		relevant_labels = ground_truth_labels[i]

		rmap = RetrievalMAP(top_k=top_k)

		mAP = rmap.update(image_scores, relevant_labels, torch.zeros(len(image_scores), dtype=torch.long))

		mAP_values.append(rmap.compute().item())

	return mAP_values

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


def to_cuda_recursive(obj):
	print(f"Type: {type(obj)}")
	if isinstance(obj, torch.Tensor):
		# Move the tensor to the CUDA device
		return obj.to('cuda')
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
		return torch.from_numpy(obj).to('cuda')
	else:
		return obj  # Return unchanged if not a tensor, list, tuple, or dict


