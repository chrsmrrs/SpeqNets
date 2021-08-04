from itertools import product

from graph_tool.all import *
import numpy as np

g = Graph(directed=False)
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()

g.add_edge(0,1)
g.add_edge(1,2)
g.add_edge(2,3)
g.add_edge(3,0)

# TODO: Iso types.
# TODO: Add self-loops?
def compute_ktuple_graph(g, k=2):
    # New k-tuple graph.
    ktuple_graph = Graph(directed=False)

    # Create iterator over all k-tuples.
    tuples = product(g.vertices(), repeat = k)

    # Map from tuple back to node in k-tuple graph.
    tuple_to_node = {}
    # Inverse of above.
    node_to_tuple = {}

    # Create a node for each k-set.
    node_labels = {}
    for s in tuples:
        v = ktuple_graph.add_vertex()

        # TODO: Iso types.
        node_labels[v] = 42

        # Manage mappings.
        node_to_tuple[v] = s
        tuple_to_node[tuple([s[i] for i in range(0,k)])] = v

    # Iterate over nodes and add edges.
    edge_to_i = {}
    for m in ktuple_graph.vertices():
        # Get corresponding tuple.
        s = list(node_to_tuple[m])

        # Iterate over components of s.
        for i in range(0, k):
            # Node to be exchanged.
            v = s[i]

            # Iterate over neighbors of node v in original graph.
            for e in v.out_neighbors():
                # Copy tuple s.
                n = s[:]
                # Exchange node v by e.
                n[i] = e
                w = tuple_to_node[tuple(n)]

                # Insert edge.
                ktuple_graph.add_edge(m,w)
                edge_to_i[(m,w)] = i

    return (ktuple_graph, node_labels, edge_to_i)

def compute_wl(graph_db, it, node_labels, edge_to_i):
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
    for g in graph_db:
        for v in g.vertices():
            colors.append(node_labels[v])

    max_all = int(np.amax(colors) + 1)
    feature_vectors = [
        np.concatenate((feature_vectors[i], np.bincount(colors[index[0]:index[1] + 1], minlength=max_all))) for
        i, index in enumerate(graph_indices)]

    stable = False
    dim = 0

    i = 0
    while i < it:
        colors = []

        for g in graph_db:
            for v in g.vertices():
                neighbors = []

                nei = g.get_out_edges(v).tolist()
                for (_, w) in nei:
                    neighbors.append(hash(tuple([node_labels[w], edge_to_i[(v,w)]])))

                neighbors.sort()
                neighbors.append(node_labels[v])
                colors.append(hash(tuple(neighbors)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        cl = []
        for g in graph_db:
            for v in g.vertices():
                node_labels[v] = colors[q]
                q += 1
                cl.append(node_labels[v])

        max_all = int(np.amax(colors) + 1)

        feature_vectors = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for i, index in
                           enumerate(graph_indices)]

        dim_new = feature_vectors[0].shape[-1]

        if dim_new == dim:
            stable = True
            break

        i += 1

    feature_vectors = np.array(feature_vectors)

    return feature_vectors


k_tuple_graph, node_labels, edge_to_i = compute_ktuple_graph(g, k=2)
print(k_tuple_graph.num_vertices(), k_tuple_graph.num_edges())
print("Gg")
compute_wl([k_tuple_graph], 5, node_labels, edge_to_i)