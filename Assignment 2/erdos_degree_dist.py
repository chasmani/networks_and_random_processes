# SO if confused, go back and look at workbook 6

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import Counter

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

N=1000

z = 1.5

N = 1000
p = z/N
G = nx.gnp_random_graph(N, p, seed=None, directed=False)


realisations = 20

full_count = Counter()

for i in range(realisations):
	ks, Pk = degree_distribution(G)
	for degree in ks:
		full_count[degree] += Pk[degree]

degree, count = zip(*full_count.items())

count_normalised = [c/20 for c in count]

plt.figure()
plt.plot(degree,count_normalised,'bo', label='Empirical distribution')
plt.xlabel("degree, k", fontsize=16)
plt.ylabel("P(k)", fontsize=16)
plt.title("Erdos-Renyi degree distribution. z={}, N={}".format(z,N))
plt.grid(True)

from scipy.stats import poisson
pkp = poisson.pmf(ks, z)
plt.plot(ks, pkp, 'ro', label='Poisson distribution')


plt.legend()
plt.savefig('poisson.png') #save the figure into a file


plt.show()