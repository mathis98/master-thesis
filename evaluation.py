import inquirer
from subprocess import call

questions = [
	inquirer.List(
		'task',
		message= 'Select a task:',
		choices=[
			'Calculate test mAP score',
			'Test embeddings',
			'Visualize via t-SNE',
		],
	),
]

answers = inquirer.prompt(questions)

if answers['task'] == 'Calculate test mAP score':
	call(['python', 'calc_map.py'])
elif answers['task'] == 'Test embeddings':
	call(['python', 'test_embedding.py'])
elif answers['task'] == 'Visualize via t-SNE':
	print('t-SNE not yet implemented')

