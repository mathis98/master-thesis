from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

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