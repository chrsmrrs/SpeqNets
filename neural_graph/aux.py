import itertools
import os.path as osp
from itertools import combinations_with_replacement

import numpy as np
from graph_tool.all import *
from torch_geometric.datasets import TUDataset


# Compute atomic type for ordered set of vertices of graph g.
def compute_atomic_type(g, vertices, node_labels, edge_labels):
    edge_list = []

    # Loop over all pairs of vertices.
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices):
            # Check if edge or self loop.
            if g.edge(v, w):
                edge_list.append((i, j, 1, node_labels[v] + 1, node_labels[w] + 1, edge_labels[g.edge(v, w)] + 1))
            elif v == w:
                edge_list.append((i, j, 0, node_labels[v] + 1, node_labels[w] + 1, 0))
            elif not g.edge(v, w):
                edge_list.append((i, j, 2, node_labels[v] + 1, node_labels[v] + 1, 0))

    edge_list.sort()

    return hash(tuple(edge_list))


# Implementation of the (k,s)-WL.
def compute_k_s_tuple_graph_fast(g, k, s, node_labels, edge_labels,  atomic_type, atomic_counter):

    # (k,s)-tuple graph.
    k_tuple_graph = Graph(directed=False)

    # Map from tuple back to node in k-tuple graph.
    tuple_to_node = {}
    # Inverse of above.
    node_to_tuple = {}

    # Manages node_labels, i.e., atomic types.
    tuple_labels = k_tuple_graph.new_vertex_property("int")

    # True if connected multi-set has been found already.
    multiset_exists = {}

    # List of s-multisets.
    k_multisets = combinations_with_replacement(g.vertices(), r=s)

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

    # True if tuple exists in k-tuple graph.
    tuple_exists = {}

    # Generate nodes of (k,s)-graph.
    # Iterate of (k,s)-multisets.
    for ms in k_multisets:
        # Create all permutations of multiset.
        permutations = itertools.permutations(ms)

        # Iterate over permutations of multiset.
        for t in permutations:
            # Check if tuple t already exists. # TODO: Needed?
            if t not in tuple_exists:
                tuple_exists[t] = True

                # Add vertex to k-tuple graph representing tuple t.
                t_v = k_tuple_graph.add_vertex()

                # Compute atomic type.
                raw_type = compute_atomic_type(g, t, node_labels, edge_labels)

                # Atomic type seen before.
                if raw_type in atomic_type:
                    tuple_labels[t_v] = atomic_type[raw_type]
                else:  # Atomic type not seen before.
                    tuple_labels[t_v] = atomic_counter
                    atomic_type[raw_type] = atomic_counter
                    atomic_counter += 1

                # Manage mappings, back and forth.
                node_to_tuple[t_v] = t
                tuple_to_node[t] = t_v

    # Iterate over nodes and add edges.

    matrices = [[],[],[]]
    labels = []

    tuple_edge_labels = k_tuple_graph.new_edge_property("int")
    for c, m in enumerate(k_tuple_graph.vertices()):

        # Get corresponding tuple.
        t = list(node_to_tuple[m])

        labels.append(tuple_labels[m])

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
                        matrices[i].append([int(m), int(w)])

                        k_tuple_graph.add_edge(m, w)
                        tuple_edge_labels[k_tuple_graph.edge(m, w)] = i + 1
                        tuple_edge_labels[k_tuple_graph.edge(w, m)] = i + 1

        # Add self-loops, only once.
        k_tuple_graph.add_edge(m, m)
        tuple_edge_labels[k_tuple_graph.edge(m, m)] = 0

    return atomic_type, atomic_counter, matrices, labels


def read_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [float(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_multi_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [[float(j) for j in i.split(",")] for i in list(f)]
    f.closed

    return np.array(classes)


def get_dataset(dataset, multigregression=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
    TUDataset(path, name=dataset)

    if multigregression:
        return read_multi_targets(dataset)
    else:
        return read_targets(dataset)
