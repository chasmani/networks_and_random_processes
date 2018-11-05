
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def average_clustering_coefficient():


	z = np.arange(0.1, 10, 0.1)
	N=1000
	realisations = 10
	p = z/N
	CC = np.zeros((len(p), realisations))
	for i in range(realisations):
	    for index, j in enumerate(p):
	        G=nx.gnp_random_graph(N,j) # build a graph
	        c=nx.average_clustering(G)
	        CC[index,i] = c

	average_CC = np.mean(CC, axis = 1)

	plt.plot(z, average_CC, "1")
	plt.ylabel("Expected clustering coef", fontsize = 16)
	plt.xlabel("z", fontsize = 16)

	plt.title("Erdos-Renyi Clustering Coefficient against z. N={}".format(N))

	plt.savefig("clustering.png")
	plt.show()	

average_clustering_coefficient()