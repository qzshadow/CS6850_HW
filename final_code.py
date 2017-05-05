#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:27:42 2017
mid term slides code for CS6850
probability version for bilingual cascade in networks
compared it to naive counterpart


assume that the innovation product cannot be replaced by old ones 
assume that only a small number of nodes are new product and the others are all old products
author: zq32@cornell.edu tl486@cornell.edu
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
        max_idx = np.argmax(attr_val)
        if max_idx == 0:
            color_res.append('black')
        else:
            color_res.append('white')
    return color_res


def get_node_transparency(G, attr):
    node_attr = nx.get_node_attributes(G, attr)  # type: map
    trans_res = []
    for attr_val in node_attr.values():
        red = hex(int(255 - attr_val[0] * 255.0))[2:]
        redresult = ('0' * (2 - len(red))) + red
        trans_res.append('#' + (redresult * 3))
    return trans_res


def graph_naive_iterate(G, pos, steps, A, B, plot=True, debug=False):
    for t in range(steps):
        if plot == True:
            # h = int(np.sqrt(steps))
            h = 1
            plt.subplot(h, ceil(steps / h), t + 1)
            n_size = 200 if debug else 30
            nx.draw_networkx_nodes(G, pos, node_color='black', node_size=n_size, linewidths=1.5)
            nx.draw_networkx_nodes(G, pos, node_color=get_node_color(G, 'attr'), node_size=n_size)
            if debug:
                nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8)
            plt.axis('off')
        # plt.show()

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
                elif neibor_color_idx == 1:
                    G.node[idx]['temp'][1] += B
                elif neibor_color_idx == 2:
                    G.node[idx]['temp'][0] += A
                    G.node[idx]['temp'][1] += B


                    # phase 2 for every node update it's attr according to its temp
        for idx in G.nodes():
            node = G.node[idx]
            node_attr = node['temp']
            if G.node[idx]['attr'][0] == 1:
                G.node[idx]['temp'] = np.array([0, 0, 0], dtype='f')
            else:
                res = np.array([0, 0, 0], dtype='f')
                res[np.argmax(node_attr)] = 1
                G.node[idx]['attr'] = res
                G.node[idx]['temp'] = np.array([0, 0, 0], dtype='f')
    plt.savefig('naive.png')
    plt.show()


def graph_iterate(G, heri_nodes, pos, steps, A, B, plot='alpha', debug=False):
    for t in range(steps):
        if plot == 'color':
            #            h = 1
            h = int(np.sqrt(steps))
            plt.subplot(h, ceil(steps / h), t + 1)
            n_size = 200 if debug else 30
            non_heuri_nodes = list(filter(lambda x: x not in heri_nodes, G.nodes()))
            non_heuri_nodes_color = map(lambda x: get_node_color(G, 'attr')[x], non_heuri_nodes)
            heuri_nodes_color = map(lambda x: get_node_color(G, 'attr')[x], heri_nodes)
            nx.draw_networkx_nodes(G, pos, nodelist=non_heuri_nodes,
                                   node_shape='o',
                                   node_color=list(non_heuri_nodes_color), node_size=n_size)
            nx.draw_networkx_nodes(G, pos, nodelist=heri_nodes, node_color=list(heuri_nodes_color),
                                   node_shape='^', node_size=n_size)
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8)
            plt.axis('off')

        elif plot == 'alpha':
            h = int(np.sqrt(steps))
            #            h = 1
            plt.subplot(h, ceil(steps / h), t + 1)
            n_size = 200 if debug else 30
            non_heuri_nodes = list(filter(lambda x: x not in heri_nodes, G.nodes()))
            non_heuri_nodes_color = map(lambda x: get_node_transparency(G, 'attr')[x], non_heuri_nodes)
            heuri_nodes_color = map(lambda x: get_node_transparency(G, 'attr')[x], heri_nodes)
            nx.draw_networkx_nodes(G, pos, nodelist=non_heuri_nodes,
                                   node_shape='o',
                                   node_color='black',
                                   node_size=n_size,
                                   linewidths=1.5)
            nx.draw_networkx_nodes(G, pos, nodelist=non_heuri_nodes,
                                   node_shape='o',
                                   node_color=list(non_heuri_nodes_color), node_size=n_size)
            nx.draw_networkx_nodes(G, pos, nodelist=heri_nodes,
                                   node_shape='^',
                                   node_color='black',
                                   node_size=n_size,
                                   linewidths=1.5)
            nx.draw_networkx_nodes(G, pos, nodelist=heri_nodes, node_color=list(heuri_nodes_color),
                                   node_shape='^', node_size=n_size)
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8)
            plt.axis('off')

        elif plot == 'text':
            num_newtech, sum_newtech = summarize_graph(G, heri_nodes)
            print("step: " + repr(t) + "red_num: " + repr(num_newtech))

        if debug:
            nx.draw_networkx_labels(G, pos)
            plt.show()

        for idx in G.nodes():
            G.node[idx]['attr'] = G.node[idx]['attr'].astype(np.float32, copy=False)
            G.node[idx]['temp'] = G.node[idx]['temp'].astype(np.float32, copy=False)

            # phase 1

        for idx in G.nodes():
            node = G.node[idx]
            node_is_cont = node['is_cont']
            if node_is_cont is True:
                for neibor_idx in G.neighbors(idx):
                    neibor_node = G.node[neibor_idx]
                    neibor_attr = neibor_node['attr']
                    G.node[idx]['temp'][0] += neibor_attr[0] * A
                    G.node[idx]['temp'][1] += neibor_attr[1] * B
            elif node_is_cont is False:
                red_cont = 0
                blue_cont = 0
                for neibor_idx in G.neighbors(idx):
                    if np.argmax(G.node[neibor_idx]['attr']) == 0:
                        red_cont += 1
                    else:
                        blue_cont += 1
                    if red_cont * A >= blue_cont * B:
                        G.node[idx]['temp'] = np.array([1, 0], dtype='f')
                    else:
                        G.node[idx]['temp'] = np.array([0, 1], dtype='f')
            else:
                raise ValueError('unknown is_cont')

                # phase 2
        for idx in G.nodes():
            if G.node[idx]['is_cont'] is True:
                node = G.node[idx]
                node_temp = node['temp']

                G.node[idx]['attr'] += node_temp
                G.node[idx]['attr'] = G.node[idx]['attr'] / sum(G.node[idx]['attr'])
                G.node[idx]['temp'] = np.array([0, 0], dtype='f')
            elif G.node[idx]['is_cont'] is False:
                G.node[idx]['attr'] = G.node[idx]['temp'].copy()

    # statistics
    # if heri_nodes:
    #     plt.savefig('defender.png')
    # else:
    #     plt.savefig('continuous.png')
    if not debug:
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


def summarize_graph(G, heri_node):
    num_newtech = 0
    sum_newtech = 0
    for idx in G.nodes():
        if G.node[idx]['attr'][0] > G.node[idx]['attr'][1]:
            num_newtech += 1
        sum_newtech += G.node[idx]['attr'][0]

    return num_newtech, sum_newtech


def heuristic_nodes_gen_(G, start_idx, defender_k):
    adj_matrix = nx.adj_matrix(G).todense()
    link_matrix = adj_matrix[start_idx:][:, 0: start_idx]
    print(link_matrix.shape)


def heuristic_nodes_gen(G, start_idx, k, N):
    interEdgeCount = np.zeros((N), dtype='int')
    for node_from, node_to in G.edges():
        if node_from < 50 <= node_to:
            interEdgeCount[node_to] += 1
        elif node_from >= 50 > node_to:
            interEdgeCount[node_from] += 1
    # print(interEdgeCount)
    subInterEdgeCount = interEdgeCount[start_idx:]
    output = subInterEdgeCount.argsort()[-k:][::-1] + start_idx
    return list(output)


def helper_debug(G):
    output = []
    for node_from, node_to in G.edges():
        if node_from < 50 <= node_to:
            output.append(node_to)
        elif node_from >= 50 > node_to:
            output.append(node_from)
    return set(output)


# ==============================================================================
# H = nx.Graph()
# H.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5), (3, 6), (5, 6)])
# fix_pos = {0: (0, 0), 1: (0, 2), 2: (1, 1), 3: (2, 1), 4: (3, 2), 5: (3, 1), 6: (3, 0)}
# pos = nx.spring_layout(H, pos=fix_pos)
# a_init_nodes_idx = [0]
# init_graph(H, a_init_nodes_idx)
# J = H.copy()
# G = H.copy()
# steps = 1
# A = 2
# B = 1

# graph_naive_iterate(G=G,
#                     pos=pos,
#                     steps=steps,
#                     A=A,
#                     B=B,
#                     plot=True,
#                     debug=False)
# 
# graph_iterate(G=H,
#               heri_nodes=[],
#               pos=pos,
#               steps=steps,
#               A=A,
#               B=B,
#               plot='alpha',
#               debug=False)
# 
# turn_nodes_to_discrete(J, [3])
# graph_iterate(G=J,
#               heri_nodes=[3],
#               pos=pos,
#               steps=steps,
#               A=A,
#               B=B,
#               plot='alpha',
#               debug=False)
# ==============================================================================

def evaluate_para(p_inner, p_inter):
    n1 = 50
    p1 = p2 = p_inner
    n2 = 50
    inter_prob = p_inter
    a_init_nodes_idx = [0]
    A = 2
    B = 1
    steps = 20
    defender_k = 5

    J = generate_two_block(n1, p1, n2, p2, inter_prob)
    while not nx.is_connected(J):
        J = generate_two_block(n1, p1, n2, p2, inter_prob)

    pos = nx.spring_layout(J)

    init_graph(J, a_init_nodes_idx)

    J_local = J.copy()
    degree_heri_nodes = heuristic_nodes_gen(J_local, n1, defender_k, n1 + n2)
    turn_nodes_to_discrete(J_local, degree_heri_nodes)
    graph_iterate(J_local, degree_heri_nodes, pos, steps, A, B, 'alpha', False)
    num_newtech, sum_newtech = summarize_graph(J_local, degree_heri_nodes)

    J_local2 = J.copy()
    degree_heri_nodes2 = degree_heristic(J_local2, defender_k, nx.betweenness_centrality)
    turn_nodes_to_discrete(J_local2, degree_heri_nodes2)
    graph_iterate(J_local2, degree_heri_nodes2, pos, steps, A, B, 'alpha', False)
    num_newtech2, sum_newtech2 = summarize_graph(J_local2, degree_heri_nodes2)

    return num_newtech, num_newtech2


def main():
    p_inner_list = np.linspace(0.2, 0.8, num=4)
    p_inter_list = np.logspace(-3, -1, num=3)
    num_experiments = np.arange(3)
    res1 = np.empty(shape=(len(num_experiments), len(p_inner_list), len(p_inter_list)))
    res2 = np.empty(shape=(len(num_experiments), len(p_inner_list), len(p_inter_list)))
    for i, p_inner in enumerate(p_inner_list):
        for j, p_inter in enumerate(p_inter_list):
            for k, exper_idx in enumerate(num_experiments):
                res1[k, i, j], res2[k, i, j] = evaluate_para(p_inner, p_inter)

    print(np.mean(res1, axis=0) - np.mean(res2, axis=0))


if __name__ == '__main__':
    main()
