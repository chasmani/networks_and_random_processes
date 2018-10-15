import random

import matplotlib.pyplot as plt

## Part d
from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


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

	p = 0.65
	realisations = 500
	observations = np.zeros(realisations)
	for i in range(realisations):
		final_position = simple_random_walk(p, total_steps, 10, "closed")[-1]
		observations[i] = final_position


	scaled_observations = (observations-1) * 10/9

	
	states = [1,2,3,4,5,6,7,8,9,10]
	stat_dist = [(p/(1-p))**x for x in states]
	total_stat_dist = sum(stat_dist)
	stat_dist_norm = [x/total_stat_dist for x in stat_dist]
	plt.hist(scaled_observations, density=True, bins=10, label="Empirical density distribution")
	plt.plot(states, stat_dist_norm, 'ro', label="Theoretical stationary distribution")
	
	plt.title('Closed SRW - {} timesteps, L=10, p=0.65'.format(total_steps,p), fontsize = 16)

	plt.legend()
	plt.xlabel('x', fontsize = 20)
	plt.ylabel('frequency', fontsize = 20)

	plt.savefig('{}_steps.png'.format(total_steps))
	
	plt.show()



def plot_distribution_1_realisation_500_steps(title):


	p = 0.65

	positions = simple_random_walk(p, 500, 10, "closed")
	np_positions = np.array(positions)
	scaled_positions = (np_positions-1) * 10/9


	plt.hist(scaled_positions, density=True, label="Empirical density distribution")

	states = [1,2,3,4,5,6,7,8,9,10]
	stat_dist = [(p/(1-p))**x for x in states]
	total_stat_dist = sum(stat_dist)
	stat_dist_norm = [x/total_stat_dist for x in stat_dist]
	plt.plot(states, stat_dist_norm, 'ro',label="Theoretical stationary distribution")
	
	plt.legend(loc=2, prop={'size': 6})

	plt.title('Closed SRW. 1 realisation. 500 timesteps, L=10, p=0.65')

	plt.xlabel('x', fontsize = 10)
	plt.ylabel('frequency', fontsize = 12)
	plt.savefig('500_steps.png')

	plt.show()

plot_distribution_1_realisation_500_steps(1)

def plot_4_realisations():

	plt.rcParams['figure.figsize'] = [10, 6]

	plt.figure(1)
	plt.suptitle('Closed SRW. 4 seperate realisations. 500 timesteps, L=10, p=0.6', fontsize = 12)

	plt.subplot(221)	
	plt.axis((0,10,0,0.5))
	plot_distribution_1_realisation_500_steps("a")
	plt.subplot(222)

	plt.axis((0,10,0,0.5))

	plot_distribution_1_realisation_500_steps("b")
	plt.subplot(223)

	plt.axis((0,10,0,0.5))

	plot_distribution_1_realisation_500_steps("c")
	plt.subplot(224)

	plt.axis((0,10,0,0.5))

	plot_distribution_1_realisation_500_steps("d")

	plt.savefig("500_steps_4.png")
	plt.show()



def one_realisation(mu, sigma, tmax):

	#np.random.seed(12)


	#Generate sequences of random variables
	x_sequence = np.random.normal(mu, sigma, tmax)
	#x_sequence = np.random.randn(tmax)*sigma + mu # How 
	y_sequence = np.cumsum(x_sequence)
	z_sequence = np.exp(y_sequence)

	return y_sequence, z_sequence
	
	
def plot_y_and_z(mu, sigma, tmax):
	
	
	
	y_sequence, z_sequence = one_realisation(mu, sigma, tmax)
	
	plt.plot(range(tmax), z_sequence, label = r'$Z_n$')
	plt.plot(range(tmax), y_sequence, label = r'$Y_n$')
	plt.legend(loc = 'upper left', fontsize = 20)
	plt.xlabel('n', fontsize=20)
	plt.show()


def empirical_results(mu, sigma, tmax, realisations=500):
	
	results = np.zeros((realisations,tmax))
	
	for realisation in range(realisations):
		results[realisation,:] = one_realisation(mu,sigma,tmax)[1]

	return results


def plot_empirical_results():

	mu = 0
	sigma = 0.2
	tmax = 100
	

	plt.rcParams['figure.figsize'] = [10, 6]

	results = empirical_results(mu,sigma,tmax)
		
	empirical_averages = results.mean(axis=0)
	empirical_sds = results.std(axis=0)/np.sqrt(500)
	
	plt.figure(0)
	plt.errorbar(range(tmax), empirical_averages, yerr=empirical_sds, label = r'Emprical Average and standard deviation')
	
	plt.legend(loc="upper left")
	plt.xlabel('n', fontsize=20)
	plt.ylabel(r'$Z_n$', fontsize=20)

	plt.title(r'Emprical Average of $Z_n$. 500 realisations, $\mu_x=0$, $\sigma_x=0.2$')

	plt.savefig('Emprical Average.png')



	
	plt.show()
	


def box_plots():
	

	mu = 0
	sigma = 0.2
	tmax = 100
	

	plt.rcParams['figure.figsize'] = [10, 6]


	results = empirical_results(mu,sigma,tmax)
	
	timestep_10_results = results[:,[9]]
	
	f, (ax1, ax2) = plt.subplots(1, 2)
	
	ax1.boxplot(timestep_10_results)
	ax1.set_yscale("log")
	ax1.set_title("n=10")
		
	ax1.set_ylabel(r'$log(Z_n)$')

	ax2.boxplot(results[:,[99]])
	#ax2.boxplot(results[:,[99]], whis=100) # Plot whiskers right up to all outlying points
	ax2.set_title("n=100")	

	ax2.set_ylabel(r'$log(Z_n)$')
	ax2.set_yscale("log")


	plt.suptitle(r"Box plots of $Z_n$ at n=10 and n=100. $\mu_x=0$, $\sigma_x=0.2$")
	plt.savefig('box_plots.png')


	plt.show()


def empirical_pdf_at_timestep_original(timestep):


	mu = 0
	sigma = 0.2
	tmax = 100
	
	
	results = empirical_results(mu,sigma,tmax)
	timestep_results = results[:,[timestep-1]]
	#plt.hist(timestep_results, density=True)
	
	# KDE plot 1
	kde = stats.gaussian_kde(timestep_results.reshape(500))
	xx = np.linspace(0,10,1000)
	#plt.plot(xx, kde(xx))
	

	# Theoretical plot 1
	rv = stats.lognorm([(sigma)*(timestep**0.5)], scale=np.exp(timestep*mu))
	plt.plot(xx, rv.pdf(xx))


	# KDE plot 2
	sns.kdeplot(timestep_results.reshape(500), gridsize=10000 )
	
	x1,x2,y1,y2 = plt.axis()
	plt.axis((0,10,y1,y2))
	

	plt.show()
	
def empirical_pdf_at_timestep(timestep):


	mu = 0
	sigma = 0.2
	tmax = 100

	plt.rcParams['figure.figsize'] = [10, 6]
	
	results = empirical_results(mu,sigma,tmax, 5000)
	timestep_results = results[:,[timestep-1]]
	#plt.hist(timestep_results, density=True)
	
	# KDE plot 1
	kde = stats.gaussian_kde(timestep_results.reshape(5000))
	xx = np.linspace(0,10,1000)
	#plt.plot(xx, kde(xx))
	

	# Theoretical plot 1
	rv = stats.lognorm([(sigma)*(timestep**0.5)], scale=np.exp(timestep*mu))
	plt.plot(xx, rv.pdf(xx), label="Theoretical Distribution")

	# KDE plot 2
	sns.kdeplot(timestep_results.reshape(5000), gridsize=10000, label="Empirical KDE")
	
	if timestep > 10:
		x1,x2,y1,y2 = plt.axis()
		plt.axis((-1,10,0,0.1))

	plt.xlabel(r'$Z_n$')
	plt.ylabel("Probablity Density")
	plt.title(r"Emprical KDE and Theoretical PDF of $Z_n$ at n={}. $\mu_x=0, \sigma_x=0.2$".format(timestep))
	plt.legend()

	#plt.savefig('emprical_pdf_kde_{}.png'.format(timestep))

	plt.show()


def empirical_pdf_at_timestep_b(timestep):

	mu = 0
	sigma = 0.2
	tmax = 100


	
	results = empirical_results(mu,sigma,tmax)
	timestep_results = results[:,[timestep-1]]
	#plt.hist(timestep_results, density=True)
	
	# KDE plot 1
	kde = stats.gaussian_kde(timestep_results.reshape(500))
	xx = np.linspace(0,10,1000)
	#plt.plot(xx, kde(xx))
	
	# KDE plot 2
	sns.kdeplot(timestep_results.reshape(500), gridsize=10000 )
	
	# Theoretical plot 1
	rv = stats.lognorm([(sigma)*(timestep**0.5)], scale=np.exp(timestep*mu))
	plt.plot(xx, rv.pdf(xx))
	
	x1,x2,y1,y2 = plt.axis()
	plt.axis((0,10,0,0.4))

	plt.show()



def log_normal_pdf(z_input, mu, sigma, timestep):

	pdf_sequence = []
	
	for z in z_input:
		
		if z == 0:
			pdf_sequence.append(0)
		else:
			exp_numerator = -1*((math.log(z) - timestep*mu)**2)
			exp_denominator = 2 * timestep * sigma**2
			answer = (1/z) * 1/(sigma) * 1/((2*math.pi*timestep)**0.5)* np.exp(exp_numerator/exp_denominator)
			pdf_sequence.append(answer)
	return pdf_sequence
	
def ergodic_average():
	
	mu = 0
	sigma = 0.2
	tmax = 100

	result = one_realisation(mu,sigma,tmax)[1]
	
	ergodic_totals = np.cumsum(result)
	n_sequence = np.array(range(tmax))+1
	ergodic_averages = np.divide(ergodic_totals, n_sequence)
	
	plt.plot(range(tmax), ergodic_averages)

	plt.ylabel(r'$\bar{\mu}_N$')
	plt.xlabel("N")
	plt.title(r"Ergodic Average of $Z_n$ for 1 realisation. $\mu_x=0, \sigma_x=0.2$")
	
	plt.savefig('ergodic_average.png')	

	
	plt.show()

		
def empirical_tail(timestep):
	
	mu = -0.02
	sigma = 0.2
	tmax = 100

	results = empirical_results(mu,sigma,tmax)
	
	timestep_results = results[:,[timestep-1]].reshape(500,)
	
	data_size=len(timestep_results)

	# Set bins edges
	data_set=sorted(set(timestep_results))
	bins=np.append(data_set, data_set[-1]+1)

	# Use the histogram function to bin the data
	counts, bin_edges = np.histogram(timestep_results, bins=bins, density=False)

	counts=counts.astype(float)/data_size

	# Find the cdf
	cdf = np.cumsum(counts)

	# Plot the cdf
	plt.plot(bin_edges[0:-1], np.ones(len(cdf))-cdf,linestyle='--', marker="o", color='b', markersize=1, label="Empirical Tail")
	plt.ylim((0,1))
	plt.ylabel("1-CDF")
	plt.xlabel(r"$Z_n$")
	plt.yscale("linear")
	plt.xscale("log")  # Q _ what axis scales are most informative?
	plt.grid(True)


	rv = stats.lognorm([(sigma)*(timestep**0.5)], scale=np.exp(timestep*mu))
	plt.plot(bin_edges[0:-1], rv.sf(bin_edges[0:-1]), color="red", label="Theoretical Survival Function")
	plt.legend()

	plt.title(r"Empirical Tail of $Z_n$ at n={}. 500 realisations, $\mu_x=-0.02, \sigma_x=0.2$".format(timestep))
	plt.savefig('empirical_tail_{}.png'.format(timestep))	

	plt.show()
