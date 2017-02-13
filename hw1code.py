# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:53:13 2017

@author: zq32 @ Cornell
"""

import networkx as nx # import networkx 1.11
from itertools import permutations
from collections import Counter

G = nx.Graph() # create a undirected graph
with open('papers.lst', mode='r', encoding = 'utf-8') as fin:
    for line in fin:
        try:
            year = int(line[:4])
        except ValueError:
            #print(line) # generally, this is 19?? year
            pass
        volumn_num, conference, author_title = line[6:8],line[11:22],line[24:]
        if year >= 1985 and year <= 2005:
            try:
                author, title = author_title.split(',', maxsplit=1)
            except ValueError:
                print(author_title) # generally, this title contains comma
            author_set = set(x.strip() for x in author.split('&'))
            if len(author_set) > 1:
                G.add_edges_from(permutations(author_set,2)) # avoid coauthors with the same name
            
            
print('create graph from file done!')
print('create a undirected graph with {0} nodes and {1} edges'.format(G.number_of_nodes(), G.number_of_edges()))

def writeHead():
    with open('hw1solution.txt', mode = 'w', encoding = 'utf-8') as fout:
        fout.write('zq32\n')

def Problems1(G):
    degree_dict = nx.degree(G)
    max_degree = max(degree_dict.values())
    degree_counter = [[degree, sum(d == degree for d in degree_dict.values())] for degree in range(max_degree+1)]
    with open('hw1solution.txt', mode = 'a', encoding = 'utf-8') as fout:
        fout.writelines(["@ 1 "+" ".join([str(l[0]), str(l[1])]) +"\n" for l in degree_counter])

if __name__ == '__main__':
    writeHead()
    Problems1(G)
        
        
        
    
        
    
            
        
