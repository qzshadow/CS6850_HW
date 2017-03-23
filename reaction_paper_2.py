#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:09:36 2017
reaction paper code for CS6850
network version and probability version for bilingal cascade
assumpt that the innovation product cannot be replaced by old ones 
assumpt that only a small number of nodes are new product and the others are all old products
author: zq32 @ Cornell
"""

import networkx as nx
from matplotlib import pyplot as plt
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
def graph_naive_iterate(G, steps, plot=False, debug=False):
    for t in range(steps):
        if plot:
            h = int(np.sqrt(steps))
            plt.subplot(h,ceil(steps / h),t + 1)
            n_size = 200 if debug else 30
            nx.draw_networkx_nodes(G, pos, node_color=get_node_color(G, 'attr'), with_labels = debug, node_size=n_size)
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
                if np.array_equal(np.array([0,0,0]), neibor_attr): # if neibor is unknown point
                    continue
                else:
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
            if G.node[idx]['temp'][2] != 0:
                G.node[idx]['temp'][2] += AB
                            
                            
    # phase 2 for every node update it's attr according to its temp
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['temp']
            if np.array_equal(node_attr, np.array([0, 0 ,0])):
                continue
            elif G.node[idx]['attr'][0] == 1:
                G.node[idx]['temp'] = np.array([0,0,0])
            else:
                res = np.array([0,0,0])
                res[np.argmax(node_attr)] = 1
                G.node[idx]['attr'] = res
                G.node[idx]['temp'] = np.array([0,0,0])
    plt.show()
                
def graph_iterate(G, steps, plot=False, debug=False):
    for t in range(steps):
        if plot:
            h = int(np.sqrt(steps))
            plt.subplot(h,ceil(steps / h),t + 1)
            n_size = 200 if debug else 30
            nx.draw_networkx_nodes(G, pos, node_color=get_node_color(G, 'attr'), with_labels = debug, node_size=n_size)
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
                if np.array_equal(np.array([0,0,0], dtype='f'), neibor_attr):
                    continue
                else:
                    neibor_color_idx = np.argmax(neibor_attr)
                    if neibor_color_idx == 0:
                        G.node[idx]['temp'][0] += neibor_attr[0] * A
                        G.node[idx]['temp'][2] += neibor_attr[0] * A
                    elif neibor_color_idx == 1:
                        G.node[idx]['temp'][1] += neibor_attr[1] * B
                        G.node[idx]['temp'][2] += neibor_attr[1] * B
                    elif neibor_color_idx == 2:
                        G.node[idx]['temp'][0] += neibor_attr[2] * A
                        G.node[idx]['temp'][1] += neibor_attr[2] * B
                        G.node[idx]['temp'][2] += neibor_attr[2] * max(A, B)
            if G.node[idx]['temp'][2] - 0 > 1e-1:
                G.node[idx]['temp'][2] += AB
                
        
    # phase 2 
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['temp']
            if np.array_equal(node_attr, np.array([0, 0 ,0], dtype='f')):
                continue
            elif np.argmax(G.node[idx]['attr']) == 0 and G.node[idx]['attr'][0] != 0:
                G.node[idx]['temp'] = np.array([0,0,0], dtype='f')
            else:
                res = G.node[idx]['temp']/sum(G.node[idx]['temp'])
                G.node[idx]['attr'] = res
                G.node[idx]['temp'] = np.array([0,0,0], dtype='f')
            
        
# initialize process
A = 1
B = 2
AB = -.5

num_nodes = 15
sparse = 0.2
steps = 10

is_debug = False
is_plot = False
if debug:
    G = nx.read_gpickle('graph.pk')
else:
    while(True):
        G = nx.gnp_random_graph(num_nodes, sparse)
        if nx.is_connected(G):
            break
    for node_idx in G.nodes():
        G.node[node_idx]['attr'] = np.array([0, 0, 0])
        G.node[node_idx]['temp'] = np.array([0, 0, 0])
    #nx.set_node_attributes(G, "t", 0)
    # initialize infections
    a_init = ceil(0.1 * num_nodes)
    
    random_nodes = choice(num_nodes, a_init, replace=False)
    a_init_nodes_idx = random_nodes[:a_init]
    
    for i in a_init_nodes_idx:
        G.node[i]['attr'] = np.array([1,0,0])
    for i in G.nodes():
        if i not in a_init_nodes_idx:
            G.node[i]['attr'] = np.array([0,1,0])
        
    nx.write_gpickle(G, 'graph.pk')

pos = nx.spring_layout(G)
G2 = G.copy()
graph_naive_iterate(G, steps, is_plot, is_debug)
graph_iterate(G2, steps, is_plot, is_debug)

    
    



