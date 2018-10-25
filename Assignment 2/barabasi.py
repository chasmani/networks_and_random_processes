import collections
import random

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def barabasi_albert_graph_complete_start(n, m, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.

    A graph of ``n`` nodes is grown by attaching new nodes each with ``m``
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If ``m`` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    if m < 1 or  m >=n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
    if seed is not None:
        random.seed(seed)

    # Add m initial nodes (m0 in barabasi-speak)
    G=nx.complete_graph(m)
    G.name="barabasi_albert_graph(%s,%s)"%(n,m)
    # Target nodes for new edges
    targets=list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes=[]
    # Start adding the other n-m nodes. The first node is m.
    source=m
    while source<n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source]*m,targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = _random_subset(repeated_nodes,m)
        source += 1
    return G


def _random_subset(seq,m):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets=set()
    while len(targets)<m:
        x=random.choice(seq)
        targets.add(x)
    return targets


def plot_graph(graph):

	pos = nx.fruchterman_reingold_layout(graph);
	plt.figure(figsize=(8,8));
	plt.axis("off");
	nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="black");
	nx.draw_networkx_edges(graph, pos, alpha=0.500);
	nx.draw_networkx_labels(graph, pos, font_color="white");
	plt.show();


def np_playground():

	data_1 = np.array(
		[[1,2,3,4,5],[2,2,2,2,2]])
	data_2 = np.array(
		[[1,2,3,4,5,6],[1,1,1,1,1,1]])

	combined_data = np.concatenate((data_1, data_2), axis=1)
	average = np.mean(combined_data, axis = 0)
	deviation = np.std(combined_data, axis = 0)
	plt.plot(combined_data[0], combined_data[1])


def get_deg_dist_once_counter():


	G_test = barabasi_albert_graph_complete_start(1000,5)
	deg_dist_test = G_test.degree()

	degree_sequence = sorted([d for n, d in G_test.degree()], reverse=True)  # degree sequence
	# print "Degree sequence", degree_sequence
	degree_counter = collections.Counter(degree_sequence)
	return degree_counter
	

def plot_one_degree_distribution():

	plt.figure()
	degree_counter = get_deg_dist_once_counter()
	degree, count = zip(*degree_counter.items())
	
	plt.plot(degree, np.array(count)/sum(count))

	plt.yscale('log') # linear y scale
	plt.xscale('log')

	plt.xlabel('Degree')
	plt.ylabel('Frequency')

	plt.show()


def get_deg_dist_times(n):

	full_data = collections.Counter()
	for count in range(n):
		full_data += get_deg_dist_once_counter()
	print(full_data)
	return full_data

def plot_20_degree_distributions():

	
	
	plt.figure()	

	# 1	
	degree_counter = get_deg_dist_once_counter()
	degree, count = zip(*degree_counter.items())
	plt.scatter(degree, np.array(count)/sum(count), color="green")

	# 20
	degree_counter_20 = get_deg_dist_times(20)
	#print(sorted(degree_counter.items(), key=lambda k: -k[1]))
	deg_data_20, count_data_20 = zip(*degree_counter_20.items())
	
	degree_20 = np.array(deg_data_20)
	count_20 = np.array(count_data_20)
	plt.scatter(degree_20, count_20/sum(count_20), color="yellow")

	# Theoretical
	theoretical = [d**-2 for d in deg_data_20]
	plt.plot(degree_20, theoretical, color="red")


	plt.yscale('log') # linear y scale
	plt.xscale('log')

	plt.xlabel('Degree')
	plt.ylabel('Frequency')



	plt.show()

plot_20_degree_distributions()




