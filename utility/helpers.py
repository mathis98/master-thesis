from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import torch

def closest_indices(embeddings):

	cosine_similarities = cosine_similarity(embeddings, embeddings)
	np.fill_diagonal(cosine_similarities, -np.inf)

	top_indices = np.unravel_index(np.argpartition(cosine_similarities, -5, axis=None)[-5:], cosine_similarities.shape)
	pairs = list(zip(top_indices[0], top_indices[1]))

	return pairs


def visualize_augmentations(data, number):

	fig, axes = plt.subplots(number, 2, figsize=(8,10))

	column_titles = ["First", "Second"]

	for row in range(number):

		images = data[row]
		original = images[0].permute(1,2,0).numpy()
		augmented = images[1].permute(1,2,0).numpy()

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


def calculate_mAP(image_embeddings, caption_embeddings, ground_truth_labels):
	mAP_values = []

	for i in range(image_embeddings.shape[0]):
		caption_embedding = caption_embeddings[i]
		
		image_scores = np.dot(image_embeddings, caption_embedding)

		relevant_labels = ground_truth_labels[i]

		ranked_indices = np.argsort(image_scores)[::-1]

		num_relevant_images = torch.sum(relevant_labels)

		if num_relevant_images == 0:
			AP = .0
		else:
			ranked = ranked_indices.copy()
			precision = np.cumsum(relevant_labels[ranked]) / (np.arange(len(relevant_labels)) + 1)
			AP = torch.sum(precision * relevant_labels) / num_relevant_images

		mAP_values.append(AP)

	mAP = np.mean(mAP_values)
	return mAP

