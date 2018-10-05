import random

import matplotlib.pyplot as plt

# 1.c Simple Random Walk
def get_next_position(current_position, step, length, boundary_conditions):

	if boundary_conditions == "unlimited":
		return current_position + step

	elif boundary_conditions == "periodic":
		# Subtract and then add 1 to deal with python indexing at 0 while still making use of modulus function
		return (current_position + step-1)%(length) + 1
	
	elif boundary_conditions == "reflecting":
		if current_position == length:
			return length -1
		elif current_position == 1:
			return 2	
		else:
			return current_position + step
		
	elif boundary_conditions == "closed":
		next_position = current_position + step
		if next_position > length:
			return length
		elif next_position < 1:
			return 1
		else:
			return next_position

	elif boundary_conditions == "absorbing":
		if current_position == length:
			return length
		elif current_position == 1:
			return 1
		else:
			return current_position + step


def simple_random_walk(p, total_steps, length, boundary_conditions):

	positions = [1]
	steps = []

	for i in range(total_steps):
		if random.random() < p:
			step = 1
		else:
			step = -1

		steps.append(step)
		current_position = positions[i]
		next_position = get_next_position(current_position, step, length, boundary_conditions)
		positions.append(next_position)

	return positions


def plot_distribution_500_realisations(total_steps):

	p = 0.9
	realisations = 500
	observations = []
	for i in range(realisations):
		final_position = simple_random_walk(p, total_steps, 10, "closed")[-1]
		observations.append(final_position)

	plt.hist(observations)
	plt.show()

def plot_distribution_1_realisation_500():

	p = 0.9
	positions = simple_random_walk(p, 500, 10, "closed")
	plt.hist(positions)
	plt.show()

	## TO DO - Normalise the histogram to show the frequency spent in each state, rather than total amount of steps
	## TO DO - add axis labels, legends
	## TO DO - add theoretical/analytical lines


plot_distribution_1_realisation_500()