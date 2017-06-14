"""
Python implementation of the ShRec3D algorithm
"""

from __future__ import division, print_function

import sys

import numpy as np
import numpy.linalg as npl

import networkx as nx

from visualizer import visualize


def contacts2distances(contacts):
    """ Infer distances from contact matrix
    """
    # create graph
    graph = nx.Graph()
    graph.add_nodes_from(range(contacts.shape[0]))

    for row in range(contacts.shape[0]):
        for col in range(contacts.shape[1]):
            freq = contacts[row, col]
            if freq != 0:
                graph.add_edge(col, row, weight=1/freq)

    # find shortest paths
    spath_mat = nx.floyd_warshall_numpy(graph, weight='weight')

    # create distance matrix
    distances = np.zeros(contacts.shape)
    for row in range(contacts.shape[0]):
        for col in range(contacts.shape[1]):
            if spath_mat[row, col] == float('inf'):
                distances[row, col] = 1000000
            else:
                distances[row, col] = spath_mat[row, col]

    return distances

def distances2coordinates(distances):
    """ Infer coordinates from distances
    """
    N = distances.shape[0]
    d_0 = []

    # pre-caching
    cache = {}
    for j in range(N):
        sumi = sum([distances[j, k]**2 for k in range(j+1, N)])
        cache[j] = sumi

    # compute distances from center of mass
    sum2 = sum([cache[j] for j in range(N)])
    for i in range(N):
        sum1 = cache[i] + sum([distances[j, i]**2 for j in range(i+1)])

        val = 1/N * sum1 - 1/N**2 * sum2
        d_0.append(val)

    # generate gram matrix
    gram = np.zeros(distances.shape)
    for row in range(distances.shape[0]):
        for col in range(distances.shape[1]):
            dists = d_0[row]**2 + d_0[col]**2 - distances[row, col]**2
            gram[row, col] = 1/2 * dists

    # extract coordinates from gram matrix
    coordinates = []
    vals, vecs = npl.eigh(gram)

    vals = vals[N-3:]
    vecs = vecs.T[N-3:]

    #print('eigvals:', vals) # must all be positive for PSD (positive semidefinite) matrix

    # same eigenvalues might be small -> exact embedding does not exist
    # fix by replacing all but largest 3 eigvals by 0
    # better if three largest eigvals are separated by large spectral gap

    for val, vec in zip(vals, vecs):
        coord = vec * np.sqrt(val)
        coordinates.append(coord)

    return np.array(coordinates).T

def deconstruct(coordinates, epsilon=0.2):
    """ Derive contact matrix from given set of coordinates
    """
    dimension = coordinates.shape[1]

    # get distances
    distances = np.zeros(2 * [coordinates.shape[0]])
    for row in range(coordinates.shape[0]):
        for col in range(coordinates.shape[0]):
            comp_sum = sum(
                [(coordinates[row, d] - coordinates[col, d])**2
                    for d in range(dimension)]
            )
            distances[row, col] = np.sqrt(comp_sum)

    # get contacts
    contacts = distances <= epsilon
    return contacts.astype(int)

def apply_shrec3d(contacts):
    """ Apply algorithm to data in given file
    """
    distances = contacts2distances(contacts)
    coordinates = distances2coordinates(distances)

    return coordinates


def main():
    """ Main function
    """
    if len(sys.argv) == 1:
        # simple example
        coords = np.array([
            [1.0,0,0],                  [1.0,1.0,0],
            [1.5,0,0],  [1.5,0.5,0],    [1.5,1.0,0],
            [2.0,0,0],                  [2.0,1.0,0],
            [2.5,0,0],                  [2.5,1.0,0],
            [3.0,0,0],  [3.0,0.5,0],    [3.0,1.0,0],
                        [3.5,0.5,0]
        ])

        #coords = np.array([
        #    [1,0,0],
        #    [1.5,0,0],
        #    [2,0,0],
        #    [2.5,0,0],
        #    [3,0,0]
        #])

        contacts = deconstruct(coords, epsilon=0.51)
        rec_coords = apply_shrec3d(contacts)

        visualize([
            (coords, 'original points'),
            (rec_coords, 'reconstructed points')
        ])
    else:
        fname = sys.argv[1]

        contacts = np.loadtxt(fname)
        rec_coords = apply_shrec3d(contacts)

        np.save('%s.ptcld' % fname, rec_coords)

if __name__ == '__main__':
    main()
