from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

def closest_indices(embeddings):

	cosine_similarities = cosine_similarity(embeddings, embeddings)
	np.fill_diagonal(cosine_similarities, -np.inf)

	top_indices = np.unravel_index(np.argpartition(cosine_similarities, -5, axis=None)[-5:], cosine_similarities.shape)
	pairs = list(zip(top_indices[0], top_indices[1]))

	return pairs


def visualize_augmentations(data, number, mean, std):

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

	for idx in range(number):

		sentences = data[idx]
		original = sentences[2]
		augmented = sentences[3]

		print(original, '->', augmented)

def relevant_list(labels):
	relevant_list = []

	for label in labels:
		relevants = torch.where(labels == label, 1, 0)
		relevant_list.append(relevants)

	return relevant_list


def calculate_mAP(image_embeddings, caption_embeddings, ground_truth_labels, top_k=10):
	mAP_values = []

	for i in range(image_embeddings.shape[0]):
		caption_embedding = caption_embeddings[i]
		
		image_scores = torch.matmul(image_embeddings, caption_embedding).cpu().numpy()

		relevant_labels = ground_truth_labels[i].cpu().numpy()

		ranked_indices = np.argsort(image_scores)[::-1]

		num_relevant_images = np.sum(relevant_labels)

		if num_relevant_images == 0:
			AP = .0
		else:
			ranked = ranked_indices[:top_k]
			precision = np.cumsum(relevant_labels[ranked]) / (np.arange(1, top_k+1))
			AP = np.sum(precision * relevant_labels[ranked]) / num_relevant_images

		mAP_values.append(AP)

	return mAP_values

def define_param_groups(model, weight_decay, optimizer_name):

	print([name, p for name, p in model.named_parameters()])

	def exclude_from_wd_and_adaptation(name):
		if 'bn' in name:
			return True
		if optimizer_name == 'lars' and 'bias' in name:
			return True

		param_groups = [
			{
				'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
				'weight_decay': weight_decay,
				'layer_adaptation': True,
			},
			{
				'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
				'weight_decay': 0.,
				'layer_adaptation': False,
			},
		]
		return param_groups

