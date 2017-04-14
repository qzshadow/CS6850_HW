import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import  ListedColormap
from collections import Counter
import numpy as np
from numpy.random import  choice
from math import ceil

def matrix_graph_iterate(G: nx.Graph, steps, A, B, AB, plot=False, debug=False):
    matrix_res = []
    pure_mat_res = []
    Trans_Matrix = np.array([
        [A, 0, A],
        [0, B, B],
        [A, B, max(A, B)]
    ], dtype='f')
    adj_mat = nx.to_numpy_matrix(G, nodelist=G.nodes())  # n*n
    color_mat = np.array(list(nx.get_node_attributes(G, 'attr').values()))  # n*3
    # pure_mat_res.append(color_mat)
    for t in range(steps):
        color_mat = np.linalg.multi_dot([adj_mat, color_mat, Trans_Matrix]) + np.array([0, 0, AB])
        pure_mat_res.append(color_mat)

    # neigh_attr_sum_mat = np.dot(adj_mat, color_mat)  # n*3 line i represents the sum of node i's neibor's attr
    # update = np.dot(neigh_attr_sum_mat, Trans_Matrix) + np.array([0, 0, AB])
    # print(update)
    # for t in range(steps):
    #     matrix_res.append([])
    #     if plot:
    #         pass
    #
    #     # phase 1 matrix algorithm
    #
    #
    #     for idx in G.nodes():
    #         node = G.node[idx]
    #         node_attr = node['attr']
    #         node_color_idx = np.argmax(node_attr)
    #         sum_neigh_attr = np.array([0,0,0], dtype='f')
    #         for neibor_idx in G.neighbors(idx):
    #             sum_neigh_attr += G.node[neibor_idx]['attr'] # type: np.array
    #
    #         G.node[idx]['temp'] = np.dot(sum_neigh_attr, Trans_Matrix) + np.array([0, 0, AB])
    #         matrix_res[t].append(G.node[idx]['temp'].copy())
    #
    #     # phase 2
    #     for idx in G.nodes():
    #         G.node[idx]['attr'] = G.node[idx]['temp'].copy()
    #         G.node[idx]['temp'] = np.array([0,0,0], dtype='f')

    return pure_mat_res

def naive_graph_iterate(G: nx.Graph, steps, A, B, AB, plot=False, debug=False):
    naive_res = []
    for t in range(steps):
        naive_res.append([])
        # phase 1 naive algorithm
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

            naive_res[t].append(G.node[idx]['temp'].copy())

        # phase 2
        for idx in G.nodes():
            G.node[idx]['attr'] = G.node[idx]['temp'].copy()
            G.node[idx]['temp'] = np.array([0, 0, 0], dtype='f')

    return naive_res

# def xdot(*args): return reduce(np.dot, args)
def main():
    A = 2
    B = 1
    AB = -.5
    Delta = 10.

    num_nodes = 6
    sparse = 0.2
    steps = 12

    is_debug = False
    is_plot = True
    if is_debug:
        G = nx.read_gpickle('graph.pk')
    else:
        while(True):
            G = nx.gnp_random_graph(num_nodes, sparse)
            if nx.is_connected(G):
                break
        for node_idx in G.nodes():
            G.node[node_idx]['attr'] = np.array([0,0,0], dtype='f')
            G.node[node_idx]['temp'] = np.array([0,0,0], dtype='f')

        a_init = int(ceil(0.1 * num_nodes))

        random_nodes = choice(num_nodes, a_init, replace=False)
        a_init_nodes_idx = random_nodes[:a_init]

        for i in a_init_nodes_idx:
            G.node[i]['attr'] = np.array([1,0,0],dtype='f')
        for i in G.nodes():
            if i not in a_init_nodes_idx:
                G.node[i]['attr'] = np.array([0,1,0], dtype='f')

        nx.write_gpickle(G, 'graph.pk')

    pos = nx.spring_layout(G)
    G2 = G.copy()
    pure_mat_res, matrix_met = np.array(matrix_graph_iterate(G,2, A, B, AB, True))
    naive_met = np.array(naive_graph_iterate(G2, 2, A, B, AB, True))
    print(pure_mat_res)
    print(matrix_met)
    print(naive_met)
    # print(np.array_equal(matrix_met, naive_met))


if __name__ == '__main__':
    main()