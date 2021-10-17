import math as m

import numpy as np
from scipy import sparse as sp


# Cosine normalization for a gram matrix.
def normalize_gram_matrix(gram_matrix):
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.float64)

    for i in range(0, n):
        for j in range(i, n):
            if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g

    return gram_matrix_norm


# Cosine normalization for sparse feature vectors, i.e., \ell_2 normalization.
def normalize_feature_vector(feature_vectors):
    n = feature_vectors.shape[0]

    for i in range(0, n):
        norm = sp.linalg.norm(feature_vectors[i])
        feature_vectors[i] = feature_vectors[i] / norm

    return feature_vectors

#
# def connect(graphs, k, s):
#     converted_graphs = []
#
#     for g in graphs:
#         # Create iterator over all k-tuples of the graph g.
#         tuples = product(g.vertices(), repeat=k)
#
#         # Iterate tuples and check size of connected components.
#         for c, t in enumerate(tuples):
#
#             # Check if number of components is s, then add star vertex.
#             if not g.edge(t[0], t[1]):
#                 e = g.add_edge(t[0], t[1])
#                 g.ep.edge_color[e] = 3
#
#         converted_graphs.append(g)
#
#     return converted_graphs
#
#
# # Connect vertex of k-tuples inducing subgraphs with s components by adding star vertex.
# def connect_s_tuples(graphs, k, s):
#     converted_graphs = []
#
#     for g in graphs:
#         # Create iterator over all k-tuples of the graph g.
#         tuples = product(g.vertices(), repeat=k)
#
#         # Iterate tuples and check size of connected components.
#         for c, t in enumerate(tuples):
#             # Create graph induced by tuple t.
#             vfilt = g.new_vertex_property('bool')
#             for v in t:
#                 vfilt[v] = True
#             k_graph = GraphView(g, vfilt)
#
#             # Compute number of components.
#             components, _ = graph_tool.topology.label_components(k_graph)
#             num_components = components.a.max() + 1
#
#             # Check if number of components is s, then add star vertex.
#             if num_components == s:
#                 # u_star = g.add_vertex()
#                 # g.vp.node_color[u_star] = 3
#
#                 # TODO XXXXX
#                 # TODO REMOVE
#                 e = g.add_edge(t[0], t[1])
#                 g.ep.edge_color[e] = 55
#
#                 # for v in t:
#                 #     e = g.add_edge(v, u_star)
#                 #     g.ep.edge_color[e] = 4
#
#                 # for w in v.out_neighbors():
#                 #     e = g.add_edge(w, u_star)
#                 #     g.ep.edge_color[e] = 5
#
#         converted_graphs.append(g)
#
#     return converted_graphs