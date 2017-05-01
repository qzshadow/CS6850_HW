#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:27:42 2017
reaction paper code for CS6850
network version and probability version for bilingal cascade
assumpt that the innovation product cannot be replaced by old ones 
assumpt that only a small number of nodes are new product and the others are all old products
add probability
author: zq32 @ Cornell
"""

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
import numpy as np
from numpy.random import choice
from math import ceil
import operator

"""
return the node color in a list
usefull when you cannot plot the graph
'r' -> red
'b' -> blue
'grey' -> grey
return example:
['r','r','b','grey']
"""


def get_node_color(G, attr):
    node_attr = nx.get_node_attributes(G, attr)
    color_res = []
    for attr_val in node_attr.values():
        if sum(attr_val) == 0:
            color_res.append('grey')
        else:
            max_idx = np.argmax(attr_val)
            if max_idx == 0:
                color_res.append('r')
            elif max_idx == 1:
                color_res.append('b')
            elif max_idx == 2:
                color_res.append('g')
            else:
                raise ValueError('error processing color')
    return color_res


def get_node_transparency(G, attr):
    node_attr = nx.get_node_attributes(G, attr)  # type: map
    trans_res = []
    for attr_val in node_attr.values():
        red = hex(int(255 - attr_val[0] * 255.0))[2:]
        redresult = ('0' * (2 - len(red))) + red
        trans_res.append('#' + (redresult * 3))
    return trans_res


def graph_iterate(G, heri_nodes, pos, steps, A, B, plot='alpha', debug=False):
    for t in range(steps):
        if plot == 'color':

            h = int(np.sqrt(steps))
            plt.subplot(h, ceil(steps / h), t + 1)
            n_size = 200 if debug else 30
            non_heri_nodes = list(filter(lambda x: x not in heri_nodes, G.nodes()))
            nodeColorNonHeri = map(lambda x: get_node_color(G, 'attr')[x], non_heri_nodes)
            nodeColorHeri = map(lambda x: get_node_color(G, 'attr')[x], heri_nodes)
            nx.draw_networkx_nodes(G, pos, nodelist=non_heri_nodes,
                                   node_shape='o',
                                   node_color=list(nodeColorNonHeri), node_size=n_size)
            nx.draw_networkx_nodes(G, pos, nodelist=heri_nodes, node_color=list(nodeColorHeri),
                                   node_shape='^', node_size=n_size)
            if debug:
                nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8)
            plt.axis('off')
            if debug:
                plt.show()

        elif plot == 'alpha':
            h = int(np.sqrt(steps))
            plt.subplot(h, ceil(steps / h), t + 1)
            n_size = 200 if debug else 30
            non_heri_nodes = list(filter(lambda x: x not in heri_nodes, G.nodes()))
            nodeColorNonHeri = map(lambda x: get_node_transparency(G, 'attr')[x], non_heri_nodes)
            nodeColorHeri = map(lambda x: get_node_transparency(G, 'attr')[x], heri_nodes)
            nx.draw_networkx_nodes(G, pos, nodelist=non_heri_nodes,
                                   node_shape='o',
                                   node_color=list(nodeColorNonHeri), node_size=n_size)
            nx.draw_networkx_nodes(G, pos, nodelist=heri_nodes, node_color=list(nodeColorHeri),
                                   node_shape='^', node_size=n_size)

            if debug:
                nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8)
            plt.axis('off')
            if debug:
                plt.show()

        for idx in G.nodes():
            G.node[idx]['attr'] = G.node[idx]['attr'].astype(np.float32, copy=False)
            G.node[idx]['temp'] = G.node[idx]['temp'].astype(np.float32, copy=False)


            # phase 1

        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['attr']
            node_is_cont = node['is_cont']
            if node_is_cont == True:
                #                node_color_idx = np.argmax(node_attr)
                for neibor_idx in G.neighbors(idx):
                    neibor_node = G.node[neibor_idx]
                    neibor_attr = neibor_node['attr']

                    G.node[idx]['temp'][0] += neibor_attr[0] * A

                    G.node[idx]['temp'][1] += neibor_attr[1] * B
            elif node_is_cont == False:
                red_cont = 0
                blue_cont = 0
                for neibor_idx in G.neighbors(idx):
                    if np.argmax(G.node[neibor_idx]['attr']) == 0:
                        red_cont += 1
                    else:
                        blue_cont += 1
                    if (red_cont * A >= blue_cont * B):
                        G.node[idx]['temp'] = np.array([1, 0], dtype='f')
                    else:
                        G.node[idx]['temp'] = np.array([0, 1], dtype='f')



            else:
                raise ValueError('unknown is_cont')




                # phase 2
        for idx in G.nodes():
            if G.node[idx]['is_cont'] == True:
                node = G.node[idx]
                node_temp = node['temp']

                G.node[idx]['attr'] += node_temp
                G.node[idx]['attr'] = G.node[idx]['attr'] / sum(G.node[idx]['attr'])
                G.node[idx]['temp'] = np.array([0, 0], dtype='f')
            elif G.node[idx]['is_cont'] == False:
                G.node[idx]['attr'] = G.node[idx]['temp'].copy()

    # statistics
    plt.show()


def generate_two_block(n1, p1, n2, p2, inter_prob):
    edge_list = []
    G = nx.gnp_random_graph(n1, p1)
    H = nx.gnp_random_graph(n2, p2)
    edge_prob = np.random.rand(len(G.node), len(H.node))
    for i, node_i in enumerate(G.nodes()):
        for j, node_j in enumerate(H.nodes()):
            if edge_prob[i, j] < inter_prob:
                edge_list.append((node_i, node_j + len(G.nodes())))

    J = nx.disjoint_union(G, H)  # type: nx.Graph
    J.add_edges_from(edge_list)

    return J


def turn_nodes_to_discrete(G, nodes):
    for idx in nodes:
        G.node[idx]['is_cont'] = False


def init_graph(J, a_init_nodes_idx):
    for idx in J.nodes():
        if idx in a_init_nodes_idx:
            J.node[idx]['attr'] = np.array([1, 0], dtype='f')
            J.node[idx]['temp'] = np.array([0, 0], dtype='f')
            J.node[idx]['is_cont'] = True
        else:
            J.node[idx]['attr'] = np.array([0, 1], dtype='f')
            J.node[idx]['temp'] = np.array([0, 0], dtype='f')
            J.node[idx]['is_cont'] = True


def degree_heristic(J, defender_k, centrality_type):
    return list(map(lambda x: x[0], sorted(centrality_type(J).items(),
                                           key=operator.itemgetter(1), reverse=True)[:defender_k]))


def closeness_heristic(J, defender_k):
    pass


H = nx.Graph()
H.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5)])
pos = nx.spring_layout(H)
a_init_nodes_idx = [0]
init_graph(H, a_init_nodes_idx)
steps = 9
A = 2
B = 1
graph_iterate(G=H,
              heri_nodes=[],
              pos=pos,
              steps=steps,
              A=A,
              B=B,
              plot='alpha',
              debug=False)



#    return
# n1 = 10
# p1 = 0.5
# n2 = 15
# p2 = 0.7
# inter_prob = 0.05
# a_init_nodes_idx = [0]
# A = 2
# B = 1
# steps = 49
# defender_k = 5
#
# J = generate_two_block(n1, p1, n2, p2, inter_prob)
# while (not nx.is_connected(J)):
#     J = generate_two_block(n1, p1, n2, p2, inter_prob)
#
# pos = nx.spring_layout(J)
#
# init_graph(J, a_init_nodes_idx)
#
# for centrality_type in [nx.degree_centrality, nx.closeness_centrality, nx.betweenness_centrality,
#                         nx.eigenvector_centrality]:
#     J_local = J.copy()
#     #    degree_heri_nodes = degree_heristic(J_local, defender_k, centrality_type)
#     degree_heri_nodes = []
#     #    print(degree_heri_nodes)
#     turn_nodes_to_discrete(J_local, degree_heri_nodes)
#     graph_iterate(J_local, degree_heri_nodes, pos, steps, A, B, 'alpha', False)
#     break
