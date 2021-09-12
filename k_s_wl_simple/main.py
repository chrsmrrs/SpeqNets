import itertools
import time
from itertools import product

import numpy as np
from graph_tool.all import *

import graph_generator as gen


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
            elif not g.edge(v, w):
                edge_list.append((i, j, 2))
            elif v == w:
                edge_list.append((i, j, 3))

    edge_list.sort()

    return hash(tuple(edge_list))


# Naive implementation of the (k,s)-WL.
def compute_k_s_tuple_graph(graphs, k, s):
    tupled_graphs = []
    node_labels_all = []
    edge_labels_all = []

    # Manage atomic types.
    atomic_type = {}
    atomic_counter = 0

    for g in graphs:
        # k-tuple graph.
        k_tuple_graph = Graph(directed=False)

        # Create iterator over all k-tuples of the graph g.
        tuples = product(g.vertices(), repeat=k)

        # Map from tuple back to node in k-tuple graph.
        tuple_to_node = {}
        # Inverse of above.
        node_to_tuple = {}

        # Manages node_labels, i.e., atomic types.
        node_labels = {}
        # True if tuples exsits in k-tuple graph.
        tuple_exists = {}

        # Create nodes for k-tuples.
        for c, t in enumerate(tuples):

            # Ordered set of nodes of tuple t.
            node_set = list(t)

            # Create graph induced by tuple t.
            vfilt = g.new_vertex_property('bool')
            for v in t:
                vfilt[v] = True
            k_graph = GraphView(g, vfilt)

            # Compute number of components.
            components, _ = graph_tool.topology.label_components(k_graph)
            num_components = components.a.max() + 1

            # Check if tuple t induces less than s+1 components.
            if num_components <= s:
                tuple_exists[t] = True
            else:
                tuple_exists[t] = False
                # Skip tuple.
                continue

            # Add vertex to k-tuple graph representing tuple t.
            v = k_tuple_graph.add_vertex()

            # Compute atomic type.
            raw_type = compute_atomic_type(g, node_set)

            # Atomic type seen before.
            if raw_type in atomic_type:
                node_labels[v] = atomic_type[raw_type]
            else:  # Atomic type not seen before.
                node_labels[v] = atomic_counter
                atomic_type[raw_type] = atomic_counter
                atomic_counter += 1

            # Manage mappings, back and forth.
            node_to_tuple[v] = t
            tuple_to_node[t] = v

        # Iterate over nodes and add edges.
        edge_labels = {}

        num_vertices = k_tuple_graph.vertices()
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
                    if tuple_exists[tuple(n)]:
                        w = tuple_to_node[tuple(n)]

                        # Insert edge, avoid undirected multi-edges.
                        if not k_tuple_graph.edge(w, m):
                            k_tuple_graph.add_edge(m, w)
                            edge_labels[(m, w)] = i + 1
                            edge_labels[(w, m)] = i + 1

            # Add self-loops, only once.
            k_tuple_graph.add_edge(m, m)
            edge_labels[(m, m)] = 0

        tupled_graphs.append(k_tuple_graph)
        node_labels_all.append(node_labels)
        edge_labels_all.append(edge_labels)

    return tupled_graphs, node_labels_all, edge_labels_all


# Implementation of the (2,1)-WL.
def compute_2_1_tuple_graph(graphs):
    tupled_graphs = []
    node_labels_all = []
    edge_labels_all = []

    # Manage atomic types.
    atomic_type = {}
    atomic_counter = 0

    for g in graphs:
        # (2,1)-tuple graph.
        k_tuple_graph = Graph(directed=False)

        # Map from tuple back to node in k-tuple graph.
        tuple_to_node = {}
        # Inverse of above.
        node_to_tuple = {}

        # Manages node_labels, i.e., atomic types.
        node_labels = {}
        # True if tuples exsits in k-tuple graph.
        tuple_exists = {}

        # Create nodes for 2-tuples.
        for c, (a, b) in enumerate(g.edges()):
            # Create two tuples for each edges.
            t_1 = (a, b)
            t_2 = (b, a)

            # Ordered sets of nodes for tuples t_1 and t_2.
            node_set_1 = list(t_1)
            node_set_2 = list(t_2)

            # Tupled exists.
            tuple_exists[t_1] = True
            tuple_exists[t_2] = True

            # Add vertex to k-tuple graph representing tuple t.
            v_1 = k_tuple_graph.add_vertex()
            v_2 = k_tuple_graph.add_vertex()

            # Compute atomic type.
            raw_type_1 = compute_atomic_type(g, node_set_1)
            raw_type_2 = compute_atomic_type(g, node_set_2)

            # Atomic type seen before.
            if raw_type_1 in atomic_type:
                node_labels[v_1] = atomic_type[raw_type_1]
            else:  # Atomic type not seen before.
                node_labels[v_1] = atomic_counter
                atomic_type[raw_type_1] = atomic_counter
                atomic_counter += 1

            # Atomic type seen before.
            if raw_type_2 in atomic_type:
                node_labels[v_2] = atomic_type[raw_type_2]
            else:  # Atomic type not seen before.
                node_labels[v_2] = atomic_counter
                atomic_type[raw_type_2] = atomic_counter
                atomic_counter += 1

            # Manage mappings, back and forth.
            node_to_tuple[v_1] = t_1
            tuple_to_node[t_1] = v_1

            node_to_tuple[v_2] = t_2
            tuple_to_node[t_2] = v_2

        # Create vertices for self loops.
        for c, a in enumerate(g.vertices()):
            # Create self-loop for every vertex.
            t_1 = (a, a)

            # Ordered sets of nodes for tuples t_1 and t_2.
            node_set_1 = list(t_1)
            tuple_exists[t_1] = True

            # Add vertex to k-tuple graph representing tuple t.
            v_1 = k_tuple_graph.add_vertex()

            # Compute atomic type.
            raw_type_1 = compute_atomic_type(g, node_set_1)

            # Atomic type seen before.
            if raw_type_1 in atomic_type:
                node_labels[v_1] = atomic_type[raw_type_1]
            else:  # Atomic type not seen before.
                node_labels[v_1] = atomic_counter
                atomic_type[raw_type_1] = atomic_counter
                atomic_counter += 1

            # Manage mappings, back and forth.
            node_to_tuple[v_1] = t_1
            tuple_to_node[t_1] = v_1

        # Iterate over vertices and add edges.
        edge_labels = {}
        for c, m in enumerate(k_tuple_graph.vertices()):
            # Get corresponding tuple.
            t = list(node_to_tuple[m])

            # Iterate over components of t.
            for i in range(0, 2):
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

        tupled_graphs.append(k_tuple_graph)
        node_labels_all.append(node_labels)
        edge_labels_all.append(edge_labels)

    return tupled_graphs, node_labels_all, edge_labels_all


# Implementation of the (3,1)-WL.
def compute_3_1_tuple_graph(graphs):
    tupled_graphs = []
    node_labels_all = []
    edge_labels_all = []

    # Manage atomic types.
    atomic_type = {}
    atomic_counter = 0

    for g in graphs:
        # (3,1)-tuple graph.
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
        set_exists = {}
        # List of connected 2-sets.
        two_sets = []
        # List of connected 3-sets.
        three_tuples = []

        # Generate 2-sets.
        for a in g.vertices():
            two_sets.append([a, a])

        for (a, b) in g.edges():
            two_sets.append([a, b])

        # Generate (3,1)-set
        for a, b in two_sets:
            for v in [a, b]:
                for w in v.out_neighbors():
                    set = [a, b, w]
                    set.sort()

                    # Check if set already exits to avoid duplicates.
                    if not (tuple(set) in set_exists):
                        set_exists[tuple(set)] = True

                        # Create all permutations of set.
                        perm = itertools.permutations(set)

                        # Iterate over permutations of set.
                        for t in perm:

                            if t not in tuple_exists:
                                three_tuples.append(t)
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

                set = [a, b, v]
                set.sort()
                perm = itertools.permutations(set)

                for t in perm:
                    if t not in tuple_exists:
                        three_tuples.append(t)
                        tuple_exists[t] = True

                        # Add vertex to k-tuple graph representing tuple t.
                        v_t = k_tuple_graph.add_vertex()

                        # Compute atomic type.
                        raw_type = compute_atomic_type(g, t)

                        # Atomic type seen before.
                        if raw_type in atomic_type:
                            node_labels[v_t] = atomic_type[raw_type]
                        else:  # Atomic type not seen before.
                            node_labels[v_t] = atomic_counter
                            atomic_type[raw_type] = atomic_counter
                            atomic_counter += 1

                        # Manage mappings, back and forth.
                        node_to_tuple[v_t] = t
                        tuple_to_node[t] = v_t

        # Iterate over nodes and add edges.
        edge_labels = {}

        for c, m in enumerate(k_tuple_graph.vertices()):

            # Get corresponding tuple.
            t = list(node_to_tuple[m])

            # Iterate over components of t.
            for i in range(0, 3):
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

        tupled_graphs.append(k_tuple_graph)
        node_labels_all.append(node_labels)
        edge_labels_all.append(edge_labels)

    return tupled_graphs, node_labels_all, edge_labels_all


# Implementation of the (k,1)-WL.
def compute_k_1_tuple_graph(graphs, k):
    tupled_graphs = []
    node_labels_all = []
    edge_labels_all = []

    # Manage atomic types.
    atomic_type = {}
    atomic_counter = 0

    for g in graphs:
        # (k,1)-tuple graph.
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
        set_exists = {}
        # List of connected 2-sets.
        two_sets = []
        # List of connected k-tuples.
        k_tuples = []

        # Generate 2-sets.
        for a in g.vertices():
            two_sets.append([a, a])

        for (a, b) in g.edges():
            two_sets.append([a, b])

        sets = two_sets


        # Generate (k,1)-set
        for _ in range(2, k-1):
            new_sets = []
            for set in sets:
                for v in set:
                    for w in v.out_neighbors():
                        new_set = list(set[:])
                        new_set.append(w)
                        new_set.sort()

                        # Check if set already exits to avoid duplicates.
                        if not (tuple(new_set) in set_exists):
                            set_exists[tuple(new_set)] = True
                            new_sets.append(new_set)

                    new_set = list(set[:])
                    new_set.append(v)
                    new_set.sort()

                    # Check if set already exits to avoid duplicates.
                    if not (tuple(new_set) in set_exists):
                        set_exists[tuple(new_set)] = True
                        new_sets.append(new_set)

            sets = new_sets

        # Generate (k,1)=graph.
        for set in sets:
            for v in set:
                for w in v.out_neighbors():
                    new_set = set[:]
                    new_set.append(w)
                    new_set.sort()

                    # Check if set already exits to avoid duplicates.
                    if not (tuple(new_set) in set_exists):
                        set_exists[tuple(new_set)] = True

                        # Create all permutations of set.
                        perm = itertools.permutations(new_set)

                        # Iterate over permutations of set.
                        for t in perm:

                            if t not in tuple_exists:
                                k_tuples.append(t)
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

                new_set = set[:]
                new_set.append(v)
                new_set.sort()
                perm = itertools.permutations(new_set)

                for t in perm:
                    if t not in tuple_exists:
                        k_tuples.append(t)
                        tuple_exists[t] = True

                        # Add vertex to k-tuple graph representing tuple t.
                        v_t = k_tuple_graph.add_vertex()

                        # Compute atomic type.
                        raw_type = compute_atomic_type(g, t)

                        # Atomic type seen before.
                        if raw_type in atomic_type:
                            node_labels[v_t] = atomic_type[raw_type]
                        else:  # Atomic type not seen before.
                            node_labels[v_t] = atomic_counter
                            atomic_type[raw_type] = atomic_counter
                            atomic_counter += 1

                        # Manage mappings, back and forth.
                        node_to_tuple[v_t] = t
                        tuple_to_node[t] = v_t



        # Iterate over nodes and add edges.
        edge_labels = {}

        for c, m in enumerate(k_tuple_graph.vertices()):

            # Get corresponding tuple.
            t = list(node_to_tuple[m])

            # Iterate over components of t.
            for i in range(0, 3):
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

        tupled_graphs.append(k_tuple_graph)
        node_labels_all.append(node_labels)
        edge_labels_all.append(edge_labels)

    return tupled_graphs, node_labels_all, edge_labels_all


# Simple implementation of 1-WL for edge and node-labeled graphs.
def compute_wl(graph_db, node_labels, edge_labels):
    # Create one empty feature vector for each graph.
    feature_vectors = []
    for _ in graph_db:
        feature_vectors.append(np.zeros(0, dtype=np.float64))

    offset = 0
    graph_indices = []

    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    colors = []
    for i, g in enumerate(graph_db):
        for v in g.vertices():
            colors.append(node_labels[i][v])

    max_all = int(np.amax(colors) + 1)
    feature_vectors = [
        np.concatenate((feature_vectors[i], np.bincount(colors[index[0]:index[1] + 1], minlength=max_all))) for
        i, index in enumerate(graph_indices)]

    dim = feature_vectors[0].shape[-1]

    c = 0
    while True:
        colors = []

        for i, g in enumerate(graph_db):
            for v in g.vertices():
                neighbors = []

                out_edges_v = g.get_out_edges(v).tolist()
                for (_, w) in out_edges_v:
                    neighbors.append(hash((node_labels[i][w], edge_labels[i][(v, w)])))

                neighbors.sort()
                neighbors.append(node_labels[i][v])
                colors.append(hash(tuple(neighbors)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        for i, g in enumerate(graph_db):
            for v in g.vertices():
                node_labels[i][v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for i, index in
                           enumerate(graph_indices)]

        dim_new = feature_vectors[0].shape[-1]

        if dim_new == dim:
            break
        dim = dim_new

        c += 1

    print(c)

    feature_vectors = np.array(feature_vectors)

    return feature_vectors


graphs = gen.create_gaurav_graphs(k=3)

# start = time.time()
# tupled_graphs, node_labels, edge_labels = compute_k_1_tuple_graph(graphs, 6)
# end = time.time()
# print(end - start)
#
# feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)
#
# if np.array_equal(feature_vectors[0], feature_vectors[1]):
#     print("Not distinguished.")
# else:
#     print("Distinguished.")
#
# exit()

# start = time.time()
# #tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(graphs, 3, 2)
# end = time.time()
# print(end - start)
#
# feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)
#
# if np.array_equal(feature_vectors[0], feature_vectors[1]):
#     print("Not distinguished.")
# else:
#     print("Distinguished.")
#
# exit()
#
# # graphs = connect(graphs, 2, 2)


graphs = gen.create_gaurav_graphs(k=3)
print("###")
tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(graphs, 3, 3)

position = sfdp_layout(tupled_graphs[0])
graph_draw(tupled_graphs[0], pos=position, output="cycle_1_1.pdf")

position = sfdp_layout(tupled_graphs[1])
graph_draw(tupled_graphs[1], pos=position, output="cycle_2_1.pdf")

print(tupled_graphs[0].num_vertices())
print("###")
feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)

if np.array_equal(feature_vectors[0], feature_vectors[1]):
    print("Not distinguished.")
else:
    print("Distinguished.")

exit()

# Specify.
# k = 3
# s = 1
#
# # Create the pairs.
# graphs = create_pair(k+1)
# print("###")
# tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(graphs, k, s)
# print("###")
# feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)
#
# # Draw for visual inspection.
# position = sfdp_layout(tupled_graphs[0])
# graph_draw(tupled_graphs[0], pos=position, output="g_0.pdf")
# position = sfdp_layout(tupled_graphs[1])
# graph_draw(tupled_graphs[1], pos=position, output="g_1.pdf")
#
#
#
# if np.array_equal(feature_vectors[0], feature_vectors[1]):
#     print("Not distinguished.")
# else:
#     print("Distinguished.")

#
# # Create the pairs.
# graphs = create_pair(k+1)
# print("###")
# tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(graphs, k, s+1)
# print("###")
# feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)
#
# # Draw for visual inspection.
# position = sfdp_layout(tupled_graphs[0])
# graph_draw(tupled_graphs[0], pos=position, output="g_2.pdf")
# position = sfdp_layout(tupled_graphs[1])
# graph_draw(tupled_graphs[1], pos=position, output="g_3.pdf")
#
# if np.array_equal(feature_vectors[0], feature_vectors[1]):
#     print("Not distinguished.")
# else:
#     print("Distinguished.")

# k = 2
# s = 1
# cycles = create_cycle_pair(5)
#
# position = sfdp_layout(cycles[0])
# graph_draw(cycles[0], pos=position, output="cycle_1.pdf")
#
# position = sfdp_layout(cycles[1])
# graph_draw(cycles[1], pos=position, output="cycle_2.pdf")
#
# tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(cycles, k, s)
# feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)
#
# if np.array_equal(feature_vectors[0], feature_vectors[1]):
#     print("Not distinguished.")
# else:
#     print("Distinguished.")
#
# position = sfdp_layout(tupled_graphs[0])
# graph_draw(tupled_graphs[0], pos=position, output="cycle_1_1.pdf")
#
# position = sfdp_layout(tupled_graphs[1])
# graph_draw(tupled_graphs[1], pos=position, output="cycle_2_1.pdf")


# k = 3
# s = 2
# directed = True
# cycles = create_cycle_pair(5)
#
# tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(cycles, k, s)
# feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)
#
# if np.array_equal(feature_vectors[0], feature_vectors[1]):
#     print("Not distinguished.")
# else:
#     print("Distinguished.")
#
# position = sfdp_layout(tupled_graphs[0])
# graph_draw(tupled_graphs[0], pos=position, output="cycle_1_2.pdf")
#
# position = sfdp_layout(tupled_graphs[1])
# graph_draw(tupled_graphs[1], pos=position, output="cycle_2_2.pdf")
