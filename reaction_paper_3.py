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

"""
original iterate function
A can be changed by a lot of B

"""
def graph_naive_iterate(G, steps, A, B, AB, plot=False, debug=False):
    for t in range(steps):
        if plot:
            h = int(np.sqrt(steps))
            plt.subplot(h,ceil(steps / h),t + 1)
            n_size = 200 if debug else 30
            nx.draw_networkx_nodes(G, pos, node_color=get_node_color(G, 'attr'), node_size=n_size)
            if debug:
                nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width = 0.5, alpha=0.8)
            plt.axis('off')
#            plt.show()
        
    # phase 1 for every node update it's temp according to it's neibor's attr
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['attr']
            node_color_idx = np.argmax(node_attr)
            for neibor_idx in G.neighbors(idx):
                neibor_node = G.node[neibor_idx]
                neibor_attr = neibor_node['attr']
                neibor_color_idx = np.argmax(neibor_attr)
                if neibor_color_idx == 0:
                    G.node[idx]['temp'][0] += A
                    G.node[idx]['temp'][2] += A
                elif neibor_color_idx == 1:
                    G.node[idx]['temp'][1] += B
                    G.node[idx]['temp'][2] += B
                elif neibor_color_idx == 2:
                    G.node[idx]['temp'][0] += A
                    G.node[idx]['temp'][1] += B
                    G.node[idx]['temp'][2] += max(A, B)
            G.node[idx]['temp'][2] += AB
                            
                            
    # phase 2 for every node update it's attr according to its temp
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['temp']
            if G.node[idx]['attr'][0] == 1:
                G.node[idx]['temp'] = np.array([0,0,0], dtype='f')
            else:
                res = np.array([0,0,0], dtype='f')
                res[np.argmax(node_attr)] = 1
                G.node[idx]['attr'] = res
                G.node[idx]['temp'] = np.array([0,0,0], dtype='f')
    plt.show()
                
def graph_iterate(G, steps, A, B, AB, plot=False, debug=False):
#    prob_A = []
#    node_A = []
    for t in range(steps):
        if plot:
            h = int(np.sqrt(steps))
            plt.subplot(h,ceil(steps / h),t + 1)
            n_size = 200 if debug else 30
            nx.draw_networkx_nodes(G, pos, node_color=get_node_color(G, 'attr'), node_size=n_size)
            if debug:
                nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width = 0.5, alpha=0.8)
            plt.axis('off')
            if debug:
                plt.show()
                
        for idx in G.nodes():
            G.node[idx]['attr'] = G.node[idx]['attr'].astype(np.float32, copy = False)
            G.node[idx]['temp'] = G.node[idx]['temp'].astype(np.float32, copy = False)
            
            
    # phase 1
            
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['attr']
            node_color_idx = np.argmax(node_attr)
            for neibor_idx in G.neighbors(idx):
                neibor_node = G.node[neibor_idx]
                neibor_attr = neibor_node['attr']

                G.node[idx]['temp'][0] += neibor_attr[0] * A
                G.node[idx]['temp'][2] += neibor_attr[0] * A

                G.node[idx]['temp'][1] += neibor_attr[1] * B
                G.node[idx]['temp'][2] += neibor_attr[1] * B

                G.node[idx]['temp'][0] += neibor_attr[2] * A
                G.node[idx]['temp'][1] += neibor_attr[2] * B
                G.node[idx]['temp'][2] += neibor_attr[2] * max(A, B)
            G.node[idx]['temp'][2] += AB
        
                
        
    # phase 2 
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['temp']
#            res = np.array([np.exp(x) for x in node_attr])
            #if incre_index == 1:
            #    continue
            #if (res[0]<G.node[idx]['attr'][0]):
            #    continue
            G.node[idx]['attr'] += node_attr
            G.node[idx]['attr'] = np.array([i if i >= 0 else 0 for i in G.node[idx]['attr']])
            G.node[idx]['attr'] = G.node[idx]['attr']/sum(G.node[idx]['attr'])
            G.node[idx]['temp'] = np.array([0,0,0], dtype='f')
            
    # statistics
    plt.show()
            
        
# initialize process
#==============================================================================
# A = 2
# B = 1
# AB = -.5
# Delta = 10.
# 
# num_nodes = 12
# sparse = 0.2
# steps = 12
# 
# is_debug = False
# is_plot = True
# if is_debug:
#     G = nx.read_gpickle('graph.pk')
# else:
#     while(True):
#         G = nx.gnp_random_graph(num_nodes, sparse)
#         if nx.is_connected(G):
#             break
#     for node_idx in G.nodes():
#         G.node[node_idx]['attr'] = np.array([0,0,0], dtype='f')
#         G.node[node_idx]['temp'] = np.array([0,0,0], dtype='f')
#     #nx.set_node_attributes(G, "t", 0)
#     # initialize infections
#     a_init = ceil(0.1 * num_nodes)
#     
#     random_nodes = choice(num_nodes, a_init, replace=False)
#     a_init_nodes_idx = random_nodes[:a_init]
#     
#     for i in a_init_nodes_idx:
#         G.node[i]['attr'] = np.array([1,0,0],dtype='f')
#     for i in G.nodes():
#         if i not in a_init_nodes_idx:
#             G.node[i]['attr'] = np.array([0,1,0], dtype='f')
#         
#     nx.write_gpickle(G, 'graph.pk')
# 
# pos = nx.spring_layout(G)
# G2 = G.copy()
# graph_naive_iterate(G, steps, is_plot, is_debug)
# graph_iterate(G2, steps, is_plot, is_debug)
#==============================================================================


H = nx.Graph()
H.add_edges_from([(0,1), (1,2), (0,2), (2,3), (3,4),(3,5),(4,5)])
pos = nx.spring_layout(H)
nx.draw_networkx(H, pos)
for idx in H.nodes():
    H.node[idx]['attr'] = np.array([0, 0, 0], dtype='f')
    H.node[idx]['temp'] = np.array([0, 0, 0], dtype='f')    
    
a_init_nodes_idx = [3]
for i in a_init_nodes_idx:
    H.node[i]['attr'] = np.array([1,0,0], dtype='f')
for i in H.nodes():
    if i not in a_init_nodes_idx:
        H.node[i]['attr'] = np.array([0,1,0], dtype='f')
        
A = 1.5
B = 1
AB = -1
Delta = 1.

H2 = H.copy()
steps = 8
graph_naive_iterate(H2, steps, A, B, AB, True, False)
graph_iterate(H, steps, A, B, AB, True, False)


#==============================================================================
# A_min, A_max = 1, 5
# AB_min, AB_max = 0, 5
# interval = 0.01
# xx, yy = np.meshgrid(np.arange(A_min, A_max, step=interval),
#                          np.arange(AB_min, AB_max, step=interval))
# samples = np.c_[xx.ravel(), yy.ravel()]
# Z = []
# for A, minus_AB in samples:
#     H_bak = H.copy()
#     graph_iterate(H_bak, 8, A, -minus_AB, False, False)
#     Z.append(np.all([np.argmax(H_bak.node[idx]['attr']) for idx in H.nodes()]))
# cmap_backgrounds = ListedColormap(['#FFAAAA', '#AAAAFF'])
# Z = np.array(Z)
# Z = Z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap=cmap_backgrounds)
#==============================================================================
#dists = np.array([x.dot(M).dot(x[np.newaxis].T) for x in samples])
#dists = dists.reshape(xx.shape)
#Z = np.abs(dists - distance) < 1e-1
#plt.figure()
#plt.pcolormesh(xx, yy, Z, cmap=cmap_backgrounds)
    



