def connect(graphs, k, s):
    converted_graphs = []

    for g in graphs:
        # Create iterator over all k-tuples of the graph g.
        tuples = product(g.vertices(), repeat=k)

        # Iterate tuples and check size of connected components.
        for c, t in enumerate(tuples):

            # Check if number of components is s, then add star vertex.
            if not g.edge(t[0], t[1]):
                e = g.add_edge(t[0], t[1])
                g.ep.edge_color[e] = 3

        converted_graphs.append(g)

    return converted_graphs


# Connect vertex of k-tuples inducing subgraphs with s components by adding star vertex.
def connect_s_tuples(graphs, k, s):
    converted_graphs = []

    for g in graphs:
        # Create iterator over all k-tuples of the graph g.
        tuples = product(g.vertices(), repeat=k)

        # Iterate tuples and check size of connected components.
        for c, t in enumerate(tuples):
            # Create graph induced by tuple t.
            vfilt = g.new_vertex_property('bool')
            for v in t:
                vfilt[v] = True
            k_graph = GraphView(g, vfilt)

            # Compute number of components.
            components, _ = graph_tool.topology.label_components(k_graph)
            num_components = components.a.max() + 1

            # Check if number of components is s, then add star vertex.
            if num_components == s:
                # u_star = g.add_vertex()
                # g.vp.node_color[u_star] = 3

                # TODO XXXXX
                # TODO REMOVE
                e = g.add_edge(t[0], t[1])
                g.ep.edge_color[e] = 55

                # for v in t:
                #     e = g.add_edge(v, u_star)
                #     g.ep.edge_color[e] = 4

                # for w in v.out_neighbors():
                #     e = g.add_edge(w, u_star)
                #     g.ep.edge_color[e] = 5

        converted_graphs.append(g)

    return converted_graphs