# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:53:13 2017

@author: zq32 @ Cornell
"""

import networkx as nx # import networkx 1.11
from itertools import permutations
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

G = nx.Graph() # create a undirected graph
with open('papers.lst', mode='r', encoding = 'utf-8') as fin:
    for line in fin:
        try:
            year = int(line[:4])
        except ValueError:
            # print(line) # generally, this is 19?? year
            pass
        volumn_num, conference, author_title = line[6:8],line[11:22],line[24:]
        if year >= 1985 and year <= 2005:
            try:
                author, title = author_title.split(',', maxsplit=1)
            except ValueError:
                #print(line) # generally, this title dose not contains authors or titles
                pass
            author_set = set(x.strip() for x in author.split('&'))
            if len(author_set) > 1:
                G.add_edges_from(permutations(author_set,2)) # avoid coauthors with the same name

# attributions of Graph
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
# for problem 1             
degree_dict = nx.degree(G)
max_degree = max(degree_dict.values())
degree_counter = [[degree, sum(d == degree for d in degree_dict.values())] for degree in range(max_degree+1)]
            
 # for problem 2
comp = nx.connected_components(G)
# largest_cc = max(nx.connected_components(G), key=len)
comp_len = [len(c) for c in sorted(comp, key = len, reverse = True)]
max_comp_len = comp_len[0]
sec_comp_len = comp_len[1]
comp_len_counter = Counter(comp_len)
num_cc = [[i, comp_len_counter[i]] for i in np.arange(1, sec_comp_len + 1)]

# for problem 3
node_path_len = nx.single_source_shortest_path_length(G, 'Hartmanis') #key:lenght, value:target_point
path_counter = Counter(node_path_len.values())

# for problem 4
depth_nodes = [[] for i in range(len(path_counter.keys()))]
for key, val in node_path_len.items():
    depth_nodes[val].append(key)
res = []
for j in np.arange(1,len(depth_nodes)):
    parent_set = set(depth_nodes[j-1])
    parent_num = 0
    for node in depth_nodes[j]:
        parent_num += len(set(G.neighbors(node)) & parent_set)
    res.append(parent_num/len(depth_nodes[j]))


           
print('create graph from file done!')
print('create a undirected graph with {0} nodes and {1} edges'.format(G.number_of_nodes(), G.number_of_edges()))

def writeHead():
    with open('hw1solution.txt', mode = 'w', encoding = 'utf-8') as fout:
        fout.write('zq32\n')

def Problems1():
    with open('hw1solution.txt', mode = 'a', encoding = 'utf-8') as fout:
        fout.writelines(["@ 1 "+" ".join([str(l[0]), str(l[1])]) +"\n" for l in degree_counter])

def Problem2():
    with open('hw1solution.txt', mode = 'a', encoding = 'utf-8') as fout:
        # for 2(a)
        fout.writelines("@ 2 "+" ".join([str(max_comp_len), str(num_nodes), str(max_comp_len/num_nodes)]) + "\n")     
        # for 2(b)
        fout.writelines(["@ 2 "+" ".join([str(l[0]), str(l[1])]) + "\n" for l in num_cc])
        
def Problem3():
    with open('hw1solution.txt', mode = 'a', encoding = 'utf-8') as fout:
        fout.writelines(["@ 3 " + " ".join([str(key), str(value)]) + "\n" for key, value in path_counter.items()])
        
def Problem4():
    with open('hw1solution.txt', mode = 'a', encoding = 'utf-8') as fout:
        fout.writelines(["@ 4 " + str(i) + "\n" for i in res])

def Problem1_plot():
    degree = np.array([l[0] for l in degree_counter if l[1] != 0])
    counter = np.array([l[1] for l in degree_counter if l[1] != 0])
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('degree ($j$)')
    ax.set_ylabel('num of nodes ($n_j$)')
    ax.set_title('Plot1: Scatterplot for $(log j, log n_j)$')
    ax.plot(degree, counter, 'ro')
    
def Problem2_plot():
    cc = np.array([l[0] for l in num_cc])
    n_cc = np.array([l[1] for l in num_cc])
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('size ($j$)')
    ax.set_ylabel('num of connected comp ($k_j$)')
    ax.set_title('Plot2: Scatterplot for $(log j, log k_j)$')
    ax.plot(cc, n_cc, 'ro')    
    
def Problem3_plot():
    fig, ax = plt.subplots()
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('path length ($j$)')
    ax.set_ylabel('num of nodes ($r_j$)')
    ax.set_title('Plot3: Histogram plot for $(j, r_j)$')
    ax.set_xticks(list(path_counter.keys()))
    rects = ax.bar(list(path_counter.keys()),list(path_counter.values()), facecolor='red')
    autolabel(rects, ax)
    
def Problem4_plot():
    fig, ax = plt.subplots()
    ax.set_title('Plot4: Histogram plot for $(j, p_j)$')
    ax.set_xlabel('path length ($j$)')
    ax.set_ylabel('average number ($p_j$)')
    ax.set_xticks(range(len(res) + 2))
    rects = ax.bar(np.arange(1,len(depth_nodes)), res, facecolor='red')
    autolabel(rects, ax, False)
    
def autolabel(rects, ax, integer=True):
    for rect in rects:
        height = rect.get_height()
        if(integer):
            ax.text(rect.get_x() + rect.get_width()/2, height,
                    "%d" % height, ha='center', va='bottom')
        else:
            ax.text(rect.get_x() + rect.get_width()/2, height,
                    "%.2f" % height, ha='center', va='bottom')
    

if __name__ == '__main__':
    # pass
    writeHead()
    Problems1()
    Problem1_plot()
    Problem2()
    Problem2_plot()
    Problem3()
    Problem3_plot()
    Problem4()
    Problem4_plot()
        
        
    
        
    
            
        
