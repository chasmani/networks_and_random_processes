import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats




def degree_distribution(GER): # From lecture - p(k)
    vk = dict(GER.degree())
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk



z = np.arange(0.1, 3, 0.1)
N=1000
realisations = 20
p = z/N
S1 = np.zeros((len(p), realisations))
S2 = np.zeros((len(p), realisations))

for i in range(realisations):
    for index, j in enumerate(p):
        G=nx.gnp_random_graph(N,j) # build a graph
        Components = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
        S1[index,i] = len(Components[0])
        S2[index,i] =  len(Components[1])


Expected_S1 = np.mean(S1, axis = 1)
sd_S1 = np.std(S1, axis = 1)
Expected_S2 = np.mean(S2, axis = 1)
sd_S2 = np.std(S2, axis = 1)


plt.errorbar(z, Expected_S1, yerr=sd_S1, label = "largest component size")
plt.errorbar(z, Expected_S2, yerr=sd_S2, label = "second largest component size")
plt.ylabel("Component Size", fontsize = 20)
plt.xlabel("z", fontsize = 20)
plt.title("Erdos-Renyi component sizes against z. N ={}".format(N))
plt.legend(loc = "upper left")

#plt.savefig("erdos_1000.png")
#plt.show()


def average_clustering_coefficient(G):

	
	avc = nx.average_clustering(G)



	z = np.arange(0.1, 10, 0.1)
	N=100
	realisations = 20
	p = z/N
	CC = np.zeros((len(p), realisations))
	for i in range(realisations):
	    for index, j in enumerate(p):
	        G=nx.gnp_random_graph(N,j) # build a graph
	        c=nx.average_clustering(G)
	        CC[index,i] = c

	average_CC = np.mean(CC, axis = 1)

	plt.figure(figsize = (10,5))
	plt.plot(z, average_CC, "o")
	plt.ylabel("Expected clustering coef", fontsize = 20)
	plt.xlabel("z", fontsize = 20)

	plt.show()	

average_clustering_coefficient()