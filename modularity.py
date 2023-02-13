import numpy as np
from numpy import linalg as LA
import pandas as pd
import networkx as nx
import scipy as sp

# This algorithm is taken from "The Method of Optimal Modularity" section of the PNAS paper


# Equation 2
# s is a column vector with s[i] being 1 if node i is in group 1, and -1 if node i is in group 2
# B is the modulatiry matrix, gained from equation 3
def Q(s,G,B):
    m = G.number_of_edges()
    return (1/(4*m)) * np.matmul(np.transpose(s), (np.matmul(B, s)))


# Equation 3
def modularity_matrix(G):
    # populate modularity matrix
    m = G.number_of_edges()
    n = G.number_of_nodes()
    B = np.zeros((n,n))
    A = nx.adjacency_matrix(G).toarray()
    for i in np.arange(n):
        ki = G.degree[i]
        for j in np.arange(n):
            kj = G.degree[j]
            B[i][j] = A[i][j] - ((ki*kj)/(2*m))
    return B

# Equation 4
# Note: if at any point a proposed split makes a zero or negative contribution to the total modularity, do not divide.


# Algorithm instructions:
def modularity(G):
    # Construct the modularity matrix (equation 3) for the network
    B = modularity_matrix(G)
    # Find the MMs most positive eigenvalue & corresponding eigenvector
    w,v = np.linalg.eig(B)
    ''' 
    # rank counts the number of positive values in an eigenvector
    # Don't think I actually need this, I will potentially remove it later
    rank = np.zeros(w.size)
    for i, vect in enumerate(v):
        for k in vect:
            if k >= 0:
                rank[i] += 1
    '''
    # u is the eigenvector with the most positive elements
    u = v[np.argmax(w)]
    # divides the graph according to the leading eigenvector
    s = np.ones(u.size)
    for i, j in zip(u, s):
        if i < 0: j = -1

    return Q(s,G,B)
print(modularity(nx.karate_club_graph()))

# Divide the network into two parts according to the signs of the elements of the eigenvector

# Repeat this process for each of the parts, using the generalized modularity matrix (equation 6)

# End.