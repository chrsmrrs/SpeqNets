import itertools
from itertools import product, combinations, combinations_with_replacement

import numpy as np
from graph_tool.all import *
import graph_tool as gt

from scipy import sparse as sp


# Compute atomic type for ordered set of vertices of graph g.
def compute_atomic_type(g, vertices):
    edge_list = []

    # TODO: Iterator over edges.
    # Loop over all pairs of vertices.
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices):
            # Check if edge or self loop.
            if g.edge(v, w):
                edge_list.append((i, j, 1))
            elif not g.edge(v, w) and v != w:
                edge_list.append((i, j, 2))
            elif v == w:
                edge_list.append((i, j, 3))

    edge_list.sort()

    return hash(tuple(edge_list))


# Implementation of the (k,s)-WL.
def compute_k_s_tuple_graph_fast(g, k, s):


    # Manage atomic types.
    atomic_type = {}
    atomic_counter = 0


    # (k,s)-tuple graph.
    k_tuple_graph = Graph(directed=False)

    # Map from tuple back to node in k-tuple graph.
    tuple_to_node = {}
    # Inverse of above.
    node_to_tuple = {}

    # Manages node_labels, i.e., atomic types.
    node_labels = {}
    # True if tuple exists in k-tuple graph.
    tuple_exists = {}
    # True if connected multi-set has been found already.
    multiset_exists = {}

    # List of s-multisets.
    k_multisets = combinations_with_replacement(g.vertices(), r = s)

    # Generate (k,s)-multisets.
    for _ in range(s, k):
        # Contains (k,s)-multisets extended by one vertex.
        ext_multisets = []
        # Iterate over every multiset.
        for ms in k_multisets:
            # Iterate over every element in multiset.
            for v in ms:
                # Iterate over neighbors.
                for w in v.out_neighbors():
                    # Extend multiset by neighbor w.
                    new_multiset = list(ms[:])
                    new_multiset.append(w)
                    new_multiset.sort()

                    # Check if set already exits to avoid duplicates.
                    if not (tuple(new_multiset) in multiset_exists):
                        multiset_exists[tuple(new_multiset)] = True
                        ext_multisets.append(list(new_multiset[:]))

                # Self-loop.
                new_multiset = list(ms[:])
                new_multiset.append(v)
                new_multiset.sort()

                # Check if set already exits to avoid duplicates.
                if not (tuple(new_multiset) in multiset_exists):
                    multiset_exists[tuple(new_multiset)] = True
                    ext_multisets.append(new_multiset)

        k_multisets = ext_multisets

    if k == s:
        k_multisets = list(k_multisets)
        #print(len(k_multisets))

    # Generate nodes of (k,s)-graph.
    # Iterate of (k,s)-multisets.
    for ms in k_multisets:
        # Create all permutations of multiset.
        permutations = itertools.permutations(ms)

        # Iterate over permutations of multiset.
        for t in permutations:
            # Check if tuple t aready exists. # TODO: Needed?
            if t not in tuple_exists:
                tuple_exists[t] = True

                # Add vertex to k-tuple graph representing tuple t.
                t_v = k_tuple_graph.add_vertex()

                # Compute atomic type.
                raw_type = compute_atomic_type(g, t)

                # Atomic type seen before.
                if raw_type in atomic_type:
                    node_labels[t_v] = atomic_type[raw_type]
                else:  # Atomic type not seen before.
                    node_labels[t_v] = atomic_counter
                    atomic_type[raw_type] = atomic_counter
                    atomic_counter += 1

                # Manage mappings, back and forth.
                node_to_tuple[t_v] = t
                tuple_to_node[t] = t_v

    # Iterate over nodes and add edges.
    edge_labels = {}
    for c, m in enumerate(k_tuple_graph.vertices()):

        # Get corresponding tuple.
        t = list(node_to_tuple[m])

        # Iterate over components of t.
        for i in range(0, k):
            # Node to be exchanged.
            v = t[i]

            # Iterate over neighbors of node v in the original graph.
            for ex in v.out_neighbors():
                # Copy tuple t.
                n = t[:]

                # Exchange node v by node ex in n (i.e., t).
                n[i] = ex

                # Check if tuple exists, otherwise ignore.
                if tuple(n) in tuple_exists:
                    w = tuple_to_node[tuple(n)]

                    # Insert edge, avoid undirected multi-edges.
                    if not k_tuple_graph.edge(w, m):
                        k_tuple_graph.add_edge(m, w)
                        edge_labels[(m, w)] = i + 1
                        edge_labels[(w, m)] = i + 1

        # Add self-loops, only once.
        k_tuple_graph.add_edge(m, m)
        edge_labels[(m, m)] = 0



    return k_tuple_graph, node_labels, edge_labels, node_to_tuple

