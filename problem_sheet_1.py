## 
import random


# 1.c Simple Random Walk

def simple_random_walk(p, total_steps):
	# Simple random walk with infinite length

	positions = [0]
	steps = []

	for i in range(total_steps):
		if random.random() < p:
			step = 1
		else:
			step = -1
		steps.append(step)
		current_position = positions[i]
		positions.append(current_position + step)
			
	return positions


def get_next_position(current_position, step, length, wrap_type):


	if wrap_type == "unlimited":
		return current_position + step

	elif wrap_type == "periodic":
		# Subtract and then add 1 to deal with python indexing at 0 while still making use of modulus function
		return (current_position + step-1)%(length) + 1
	
	elif wrap_type == "reflecting":
		
		if current_position == length:
			return length -1
		elif current_position == 1:
			return 2	
		else:
			return current_position + step

		
	elif wrap_type == "closed":
		next_position = current_position + step
		if next_position > length:
			return length
		elif next_position < 1:
			return 1
		else:
			return next_position

	elif wrap_type == "absorbing":
		if current_position == length:
			return length
		elif current_position == 1:
			return 1
		else:
			return current_position + step


def simple_random_walk_wrappings(p, total_steps, length, wrap_type):

	positions = [3]
	steps = []

	for i in range(total_steps):
		if random.random() < p:
			step = 1
		else:
			step = -1

		steps.append(step)
		current_position = positions[i]

		next_position = get_next_position(current_position, step, length, wrap_type)

		positions.append(next_position)

	return positions


print(simple_random_walk_wrappings(0.5, 10,5,"absorbing"))

