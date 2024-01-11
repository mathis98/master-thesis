import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


ground_truth = int(input('Query index: '))
returned = eval(input('Please give the returned dict: '))
number = int(input('How many items (default 5): ').strip() or "5")
filename = input('Please give a filename: ')

if number > 20:
	number = 20

rows = number//5
if number%5 != 0:
	rows += 1

fig, axes = plt.subplots(rows, 5, figsize=(10-(rows),(rows)+2))

fig.suptitle(f'Query: {returned["query"]} ({ground_truth})')

num_items = 100 if returned['dataset'] == 'ucm' else 700
img_path = '../Datasets/UCM/imgs' if returned['dataset'] == 'ucm' else '../Datasets/NWPU-Captions-main/NWPU-RESISC45'

idx_val = 0

for row in range(rows):
	for col in range(5):

		modifier = axes[row, col] if number > 5 else axes[col]
		idx = returned['idxs'][idx_val]

		idx_val += 1

		if returned['dataset'] == 'nwpu':
			index = idx + 1
			category_index = index // 700
			elem_index = index % 700 - 1

			categories = sorted(os.listdir(f'../{img_path}'))

			name = f' {categories[category_index]}_{elem_index}'

		path = f'../../Datasets/UCM/imgs/{idx+1}.tif' if returned['dataset'] == 'ucm' else f'../{img_path}/{categories[category_index]}/{categories[category_index]}_{elem_index}.jpg'

		modifier.axis('off')

		if idx_val-1 >= number:
			continue

		img = plt.imread(path)
		modifier.imshow(img)
		modifier.set_title(idx)
		

		color = 'g' if ground_truth // num_items == idx // num_items else 'r'
		style = 'dashed' if idx == ground_truth else 'solid'

		border = patches.Rectangle((0, 0), 1, 1, linewidth=4, linestyle=style, edgecolor=color, facecolor='none', transform=modifier.transAxes)
		modifier.add_patch(border)

plt.tight_layout()
plt.savefig(f'../../figures/{filename}.png')
plt.show()