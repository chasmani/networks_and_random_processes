import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import poisson

m0 = 5 #initial number of nodes
m = 5 #5 edges added each time (preferential attachment)
N = 100
G = nx.barabasi_albert_graph(N, m) #The initialization is a graph with with m nodes and no edges.

N = len(G)
M = G.number_of_edges()
print('Number of nodes:', N)
print('Number of edges:', M)
print('Average degre:', 2*M/N)

pos = nx.fruchterman_reingold_layout(G);
plt.figure(figsize=(8,8));
plt.axis("off");
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="black");
nx.draw_networkx_edges(G, pos, alpha=0.500);
nx.draw_networkx_labels(G, pos, font_color="white");

def degree_distribution(GER):
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

Count_degree = list(dict(G.degree()).values())

def one_cdf(data):

    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)
    return bin_edges[0:-1], np.ones(len(cdf))-cdf


ks, Pk = degree_distribution(G)

plt.figure()
plt.plot(ks,Pk,'bo', label='Data')
plt.xlabel("k", fontsize=20)
plt.ylabel("P(k)", fontsize=20)
plt.title("Degree distribution", fontsize=20)
plt.grid(True)

plt.figure()
cdf = np.cumsum(Pk)
x = np.linspace(np.min(Count_degree),np.max(Count_degree),10000)
plt.plot(range(len(cdf)), np.ones(len(cdf))-cdf, 'o', label ='One realisation, empirical tail ')
plt.plot(x, 20*x**(-2), label = 'power law')
plt.xlabel('degree, k', fontsize = 16)
plt.title('Empirical Tail (1 realisation)', fontsize = 16)
plt.xlim([np.min(Count_degree),np.max(Count_degree)])
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize = 12)

realisations = 20
m = 5
N =1000
degrees = []
for r in range(realisations):
    G= nx.barabasi_albert_graph(N,m)
    Count_degree = list(dict(G.degree()).values())
    degrees.append(Count_degree)
degrees = np.array(degrees)
degrees = degrees.flatten()


plt.figure()
cdf = np.cumsum(Pk)
x = np.linspace(np.min(Count_degree),np.max(Count_degree),10000)
plt.plot(range(len(cdf)), np.ones(len(cdf))-cdf, '.', label ='One realisation, empirical tail ')
plt.xlabel('degree, k', fontsize = 16)
plt.title('Empirical Tail (1 realisation)', fontsize = 16)
plt.xlim([np.min(Count_degree),np.max(Count_degree)])

bins, deg = one_cdf(degrees)
plt.plot(bins, deg, '.', label = '20 realisations, empirical tail', )
Xx = np.linspace(min(degrees), max(degrees), 1000)
plt.plot(Xx, 20*Xx**(-2), label = 'Power law')
plt.title('Barabási–Albert Empirical Tail. $m=5$', fontsize = 16)

plt.legend(fontsize =12)
plt.xlabel('degree, k', fontsize = 16)
plt.ylabel('1 - PDF', fontsize = 16)
plt.xscale('log')
plt.yscale('log')
plt.xlim([np.min(degrees),np.max(degrees)])
plt.ylim([1e-4, 10])


plt.savefig("barabasi.png")


plt.show()



m = 10
N = 1000
G = nx.barabasi_albert_graph(N, m) 


def k_nn(G,kmax):
    knn = np.zeros(kmax) #be careful if you get zeros! There are undefined ( can't divide through by zero)
    #knn = -1 & np.ones(kmax) #be careful if you get zeros! There are undefined ( can't divide through by zero)
    Degrees = list(dict(G.degree()).values())
    Degrees = np.array(Degrees)
    knn_i = np.array(list(nx.average_neighbor_degree(G).values()))
    for k in range(kmax):
        delta_ki_k = np.equal(Degrees, k).astype(int)
        numerator = np.sum(delta_ki_k*knn_i)
        denominator = np.sum(delta_ki_k)
        if denominator !=0:
            knn[k] = numerator/denominator
    return knn



realisations = 100
kmax = 200
M = np.zeros((kmax, realisations))
for i in range(realisations):
    G = nx.barabasi_albert_graph(N,m)
    knn_final = k_nn(G, kmax)
    for k in range(len(knn_final)):
        if knn_final[k] == 0:
        #if knn_final[k] == -1:
            knn_final[k] = None #undefined values
    M[:,i] = knn_final


Average = np.nanmean(M, axis = 1) # If any realisation have NaN in it, it will give a Nan, so losing alot of data


plt.figure(figsize = (5,5))
print()
plt.plot(range(kmax), Average, 'x')
plt.xlim([8,kmax])
plt.ylim([28,40])

#plt.plot(range(kmax), better_Average, 'x')
plt.xlabel('k, Degree of node', fontsize = 16)
plt.ylabel(r'$k_{nn}(k)$ ', fontsize = 16)

plt.title("Barabási–Albert m=5\nDistribution of degree of nearest neighbour.")

plt.savefig('degree_neighbour.png')


plt.show()
















