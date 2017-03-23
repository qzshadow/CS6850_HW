# -*- coding: utf-8 -*-
"""
reaction paper code for CS6850
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
contains bugs

"""
def graph_naive_iterate(G, steps, plot=False):
    pos = nx.spring_layout(G)
    for t in range(steps):
                if plot:
            h = int(np.sqrt(steps))
            plt.subplot(h, ceil(step / h), t + 1)
            nx.draw_networkx(G, pos, node_color=get_node_color(G, 'attr'))
            plt.axis('off')
            plt.show()
        
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
                        if node_color_idx == 0 or node_color_idx == 2:
                            G.node[idx]['temp'][0] += A
                        elif node_color_idx == 1:
                            pass
                        else:
                            raise ValueError
                    elif neibor_color_idx == 1:
                        if node_color_idx == 1 or node_color_idx == 2:
                            G.node[idx]['temp'][1] += B
                        elif node_color_idx == 0:
                            pass
                        else:
                            raise ValueError
                    elif neibor_color_idx == 2:
                        if node_color_idx == 0:
                            G.node[idx]['temp'][0] += A
                        elif node_color_idx == 1:
                            G.node[idx]['temp'][1] += B
                        elif node_color_idx == 2:
                            G.node[idx]['temp'][2] += max(A, B)
            if G.node[idx]['temp'][2] != 0:
                G.node[idx]['temp'][2] += AB
                            
                            
    # phase 2 for every node update it's attr according to its temp
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['temp']
            if np.array_equal(node_attr, np.array([0, 0 ,0])):
                continue
            else:
                res = np.array([0,0,0])
                G.node[idx]['temp'] = np.array([0,0,0])
                res[np.argmax(node_attr)] = 1
                G.node[idx]['attr'] = res
                
        
# initialize process
A = 2
B = 1
AB = 0


num_nodes = 6

while(True):
    G = nx.gnp_random_graph(num_nodes, 0.3)
    if nx.is_connected(G):
        break
for node_idx in G.nodes():
    G.node[node_idx]['attr'] = np.array([0, 0, 0])
    G.node[node_idx]['temp'] = np.array([0, 0, 0])
#nx.set_node_attributes(G, "t", 0)
# initialize infections
a_init = ceil(0.1 * num_nodes)
b_init = ceil(0.1 * num_nodes)

steps = 10
random_nodes = choice(num_nodes, a_init + b_init, replace=False)
a_init_nodes_idx = random_nodes[:a_init]
b_init_nodes_idx = random_nodes[a_init:a_init+b_init]

for i in a_init_nodes_idx:
    G.node[i]['attr'] = np.array([1,0,0])
for i in b_init_nodes_idx:
    G.node[i]['attr'] = np.array([0,1,0])

graph_naive_iterate(G, steps, True)

    
    



