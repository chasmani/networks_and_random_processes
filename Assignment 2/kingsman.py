import numpy as np
import matplotlib.pyplot as plt

# sample paths of the process (2.1 d)

# Normalised version
#plt.plot([time,time+2*waitTime],[1/L,1/L],'r--') # also add this line

# Normalise it, so start at 1 and reduce to 0. Then can plot several different Ls on the same graph
# Also add a deterministic function to overlay on the top


COLORS = {
	10:'r',
	100:'g',
	1000:'b'
}

def plot_kingsman(L, normalised=False):

	##### Simulate and PLOT ####

	time=0.0 # initialize

	for n in range(L,1,-1):  
		rate = n*(n-1)/2 # r(n, n-1) = n choose 2. Rate of moving states
		beta=1.0/rate # beta. Average waiting time in a state
		
		# Beta is the scale function in the exponential distribution
		waitTime=np.random.exponential(scale=beta) # get a waiting time from exponential dist

		# Want this to be a strainght line, so plot a bit at a time
		# Each loop plots the next part of the time
		#plt.plot([time,time+waitTime],[n,n],'r',lw=2) # plot a bit
		
		if normalised:
			y = n/L
		else:
			y = n

		# Normalised version
		plt.plot([time,time+waitTime],[y,y],COLORS[L],lw=2) # plot a bit
		
		time+=waitTime # update time
		

	plt.plot([time,time],[y,y],COLORS[L],lw=2, label="L={}".format(L)) # Just add the legend labels
		

def plot_kingsman_multiple_normalised():

	plot_kingsman(10, normalised=True)
	plot_kingsman(100, normalised=True)
	plot_kingsman(1000, normalised=True)
		
	plt.title(r'Kingmans Coalesent for $L=10,100,1000$')
	plt.xlabel('$t$')
	plt.ylabel('$N_t$')

	plt.yscale('linear') # linear y scale
	plt.xscale('log') # change to log x scale]

	plt.legend()

	plt.show()


def plot_kingsman_multiple():

	plot_kingsman(10)
	plot_kingsman(100)
	plot_kingsman(1000)
		
	plt.title(r'Kingmans Coalesent for $L=10,100,1000$')
	plt.xlabel('$t$')
	plt.ylabel('$N_t$')

	plt.yscale('linear') # linear y scale
	plt.xscale('log') # change to log x scale]

	plt.legend()

	plt.show()

plot_kingsman_multiple_normalised()

