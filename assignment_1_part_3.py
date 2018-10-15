import random

import matplotlib.pyplot as plt

from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 6]


def simulate_results(max_time, L):

	np.random.seed(1234)

	X = np.zeros(shape = (max_time, L)) #matrix, each row is new generation
	X[0,:] = range(L) # initially individual i has type i (going from 0 - 99)

	for t in range(1, max_time):
	    old_states = X[t-1,:]
	    new_states = [old_states[r] for r in np.random.randint(0,L, L)]
	    X[t,:] = np.sort(new_states) #try it without np.sort
	#     X[t,:] = new_states
	return X


def simulation_1():

	max_time = 400
	L = 100

	X = simulate_results(max_time, L)
	
	plt.figure()
	plt.ylabel('Generation, t', fontsize = 20)
	plt.xlabel(r'Individuals, $X$', fontsize = 20)
	plt.title(r'Wright-Fisher Dynamics. $L={}$, $T={}$'.format(L,max_time))
	plt.pcolor(X)


	plt.show()


def simulation_2():

	max_time = 1000
	L = 100
	realisations = 10

	for realisation in realisations:

		X = simulate_results(max_time, L)
		unique_counts = np.zeros(max_time)
		for t in range(max_time):		
			unique_counts[t]=(len(np.unique(X[t,:])))

	
		print(X.shape)


	
	
	plt.figure()
	plt.plot(range(max_time), unique_counts)
	plt.ylabel(r'Surviving Types', fontsize = 20)
	plt.xlabel(r'Generation, t', fontsize = 20)
	plt.title(r'Wright-Fisher Dynamics. $L={}$, $T={}$'.format(L,max_time), fontsize=22)
	#plt.savefig('Wright-Fisher_trend.png')	

	plt.xscale("log")



	plt.show()
	"""
	
	sns.set_palette("Set1", 8, .75)

	plt.figure()
	pcm = sns.heatmap(X,cbar_kws={'label': r'Individual $j \in U[0,{}]$'.format(L)})
	pcm.figure.axes[-1].yaxis.label.set_size(16)
	plt.ylabel('Generation, t', fontsize = 20)
	plt.xlabel(r'Individuals, $X$', fontsize = 20)
	plt.title(r'Wright-Fisher Dynamics. $L={}$, $T={}$'.format(L,max_time), fontsize=22)
	#plt.locator_params(axis='y', nbins=50)

	plt.savefig('Wright-Fisher-70.png')	
	plt.show()
	
	"""



def am_done(v):
    '''Return true if all elements of v the same. Else false.'''
    u=np.unique(v) # unique  elements of v
    if u.shape[0]>1:
        return False
    else:
        return True


def average_lifetime():

	L = 200

	array_L = np.arange(1,L)

	max_time = 800
	repeat = 20
	M = np.zeros((len(array_L), repeat))
	for i in range(1, len(array_L)):
	    for j in range(0, repeat):
	        X=np.zeros((max_time+1,array_L[i])) #Initialize
	        X[0,:]=np.arange(array_L[i]) # at t=0 individual i has type i

	        for t in range(1,max_time+1): # Time steps. list of integers from 1 to T.

	            old_states=X[t-1,:]
	            new_states = [old_states[r] for r in np.random.randint(0,array_L[i], array_L[i])]


	            new_states=np.sort(new_states)

	            X[t,:]=new_states

	            if am_done(new_states):
	                M[i,j]=t-1
	                break



	average = np.mean(M, axis = 1)
	deviation = np.std(M, axis = 1)/np.sqrt(repeat)

	plt.ylabel(r'Steps, n', fontsize = 20)
	plt.xlabel(r'Number of Individuals, L', fontsize = 20)
	plt.title(r'Wright-Fisher steps to stationary distribution.', fontsize = 20)


	reciprocals = []

	for sd in deviation:
		if sd>0.1:
			reciprocals.append(1/sd)
		else:
			reciprocals.append(100)

	# Weight to 1/sd
	
	fit = np.polyfit(array_L, average,1,w=reciprocals)
	fit_fn = np.poly1d(fit)

	print(fit)

	plt.plot(array_L, fit_fn(array_L), label=r"Linear fit n={:+.2f}L".format(fit[0]))
	
	plt.errorbar(array_L, average, yerr = deviation, fmt = 'o', markersize=2, mew = 4, label = 'steps (n) until stationary distribution \nwith 1 std')
	plt.legend(loc='upper left')
	plt.grid()

	plt.savefig("Wright_Fisher_fixation")

	plt.show()

average_lifetime()