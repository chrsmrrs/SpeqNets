from itertools import product

import numpy as np
from graph_tool.all import *


# Create cycle counter examples.
def create_pair(k):
    # Graph 1.
    # First cycle.
    c_1 = Graph(directed=False)

    for i in range(0, k + 1):
        c_1.add_vertex()

    for i in range(0, k + 1):
        c_1.add_edge(i, (i + 1) % (k + 1))

    # Second cycle.
    c_2 = Graph(directed=False)
    for i in range(0, k + 1):
        c_2.add_vertex()

    for i in range(0, k + 1):
        c_2.add_edge(i, (i + 1) % (k + 1))

    cycle_union_1 = graph_union(c_1, c_2)
    cycle_union_1.add_edge(0, k + 1)

    c_3 = Graph(directed=False)
    for i in range(0, k + 2):
        c_3.add_vertex()

    for i in range(0, k + 2):
        c_3.add_edge(i, (i + 1) % (k + 2))

    c_4 = Graph(directed=False)
    for i in range(0, k + 2):
        c_4.add_vertex()

    for i in range(0, k + 1):
        c_4.add_edge(i, (i + 1))

    merge = c_4.new_vertex_property("int")
    for v in c_4.vertices():
        merge[v] = -1

    merge[0] = 0
    merge[k + 1] = 1

    cycle_union_2 = graph_union(c_3, c_4, intersection=merge)

    #Draw for visual inspection.
    position = sfdp_layout(cycle_union_1)
    graph_draw(cycle_union_1, pos=position, output="g_1.pdf")
    position = sfdp_layout(cycle_union_2)
    graph_draw(cycle_union_2, pos=position, output="g_2.pdf")

    return (cycle_union_1, cycle_union_2)


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
            if v == w:
                edge_list.append((i, j, 2))

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
        for t in tuples:
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
        for m in k_tuple_graph.vertices():
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

                    # Check if tuple exists.
                    if tuple_exists[tuple(n)]:
                        w = tuple_to_node[tuple(n)]

                        # Insert edge, avoid unidirected multi edges.
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
                    neighbors.append(hash(tuple([node_labels[i][w], edge_labels[i][(v, w)]])))

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


# Specify.
k = 4
s = 1

# Create the pairs.
graphs = create_pair(k+1)
print("###")
tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(graphs, k, s)
print("###")
feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)

if np.array_equal(feature_vectors[0], feature_vectors[1]):
    print("Not distinguished.")
else:
    print("Distinguished.")


# Create the pairs.
graphs = create_pair(k+1)
print("###")
tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(graphs, k, s+1)
print("###")
feature_vectors = compute_wl(tupled_graphs, node_labels, edge_labels)

if np.array_equal(feature_vectors[0], feature_vectors[1]):
    print("Not distinguished.")
else:
    print("Distinguished.")


