from itertools import product, combinations

import numpy as np
from graph_tool.all import *


def even_subsets(s):
    n = len(s)
    for r in range(1, n + 1):
        if r % 2 == 0:
            for combo in combinations(s, r):
                yield (combo)


def odd_subsets(s):
    n = len(s)
    for r in range(1, n + 1):
        if r % 2 != 0:
            for combo in combinations(s, r):
                yield (combo)


def create_gaurav_graphs(k):
    # Create complete graph K on k+1 vertices.
    K = Graph(directed=False)
    for i in range(k + 1):
        K.add_vertex()

    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            K.add_edge(i, j)

    G = Graph(directed=False)

    edge_to_e = {}
    v_S_nodes = []

    # Add vertices to graph G.
    for v in range(k + 1):
        # Neighbors of vertex v.
        E_v = list(range(0, k + 1))
        E_v.remove(v)
        temp = [(v,e) for e in E_v]
        E_v = temp

        # Add even subset vertices.
        for S in even_subsets(E_v):
            w = G.add_vertex()
            v_S_nodes.append((w, v, S))

        # Add nodes e_0, and e_1
        for e in E_v:
            e_0 = G.add_vertex()
            e_1 = G.add_vertex()

            edge_to_e[(v, e[1])] = (e_0, e_1)
            edge_to_e[(e[1], v)] = (e_0, e_1)

            G.add_edge(e_0, e_1)

    for w, v, S in v_S_nodes:
        for e in K.edges():
            u, v = e
            u = int(u)
            v = int(v)

            if v in list(e) and ((v,w) in S or (w,v) in S):
                G.add_edge(w, edge_to_e[(int(u), int(v))][1])
            elif v in list(e):
                G.add_edge(w, edge_to_e[(int(u), int(v))][0])
            else:
                print("ERROR")
                exit()

    #############################
    H = Graph(directed=False)

    edge_to_e = {}
    v_S_nodes = []

    # Add vertices to graph G.
    for v in range(k + 1):
        # Neighbors of vertex v.
        E_v = list(range(0, k + 1))
        E_v.remove(v)
        temp = [(v,e) for e in E_v]
        E_v = temp

        if v == 0:
            # Add even subset vertices.
            for S in odd_subsets(E_v):
                w = H.add_vertex()
                v_S_nodes.append((w, v, S))
        else:
            # Add even subset vertices.
            for S in even_subsets(E_v):
                w = H.add_vertex()
                v_S_nodes.append((w, v, S))

        # Add nodes e_0, and e_1
        for e in E_v:
            e_0 = H.add_vertex()
            e_1 = H.add_vertex()

            edge_to_e[(v, e[1])] = (e_0, e_1)
            edge_to_e[(e[1], v)] = (e_0, e_1)

            H.add_edge(e_0, e_1)

    for w, v, S in v_S_nodes:
        for e in K.edges():
            u, v = e
            u = int(u)
            v = int(v)

            if v in list(e) and ((v,w) in S or (w,v) in S):
                H.add_edge(w, edge_to_e[(int(u), int(v))][1])
            elif v in list(e) and ((v,w) not in S and (w,v) not in S):
                H.add_edge(w, edge_to_e[(int(u), int(v))][0])
            else:
                print("ERROR")
                exit()

    return (G,H)


def create_cfi_2_graphs():
    g1 = Graph(directed=False)
    for i in range(0, 40):
        g1.add_vertex()

    g1.add_edge(0, 4)
    g1.add_edge(0, 6)
    g1.add_edge(0, 9)
    g1.add_edge(1, 5)
    g1.add_edge(1, 7)
    g1.add_edge(1, 9)
    g1.add_edge(2, 4)
    g1.add_edge(2, 7)
    g1.add_edge(2, 8)
    g1.add_edge(3, 5)
    g1.add_edge(3, 6)
    g1.add_edge(3, 8)
    g1.add_edge(4, 20)
    g1.add_edge(5, 21)
    g1.add_edge(6, 10)
    g1.add_edge(7, 11)
    g1.add_edge(8, 30)
    g1.add_edge(9, 31)
    g1.add_edge(10, 12)
    g1.add_edge(10, 15)
    g1.add_edge(11, 13)
    g1.add_edge(11, 14)
    g1.add_edge(12, 18)
    g1.add_edge(12, 17)
    g1.add_edge(13, 16)
    g1.add_edge(13, 17)
    g1.add_edge(14, 18)
    g1.add_edge(14, 19)
    g1.add_edge(15, 16)
    g1.add_edge(15, 19)
    g1.add_edge(16, 23)
    g1.add_edge(17, 32)
    g1.add_edge(18, 25)
    g1.add_edge(19, 34)
    g1.add_edge(20, 24)
    g1.add_edge(20, 28)
    g1.add_edge(21, 22)
    g1.add_edge(21, 26)
    g1.add_edge(23, 24)
    g1.add_edge(23, 26)
    g1.add_edge(25, 22)
    g1.add_edge(25, 28)
    g1.add_edge(27, 22)
    g1.add_edge(27, 24)
    g1.add_edge(29, 26)
    g1.add_edge(29, 28)
    g1.add_edge(31, 33)
    g1.add_edge(31, 36)
    g1.add_edge(30, 37)
    g1.add_edge(30, 39)
    g1.add_edge(32, 33)
    g1.add_edge(32, 39)
    g1.add_edge(34, 37)
    g1.add_edge(34, 35)
    g1.add_edge(36, 37)
    g1.add_edge(36, 39)
    g1.add_edge(38, 33)
    g1.add_edge(38, 35)
    g1.add_edge(27, 36)
    g1.add_edge(29, 28)
    g1.add_edge(29, 26)
    g1.add_edge(29, 38)

    g2 = Graph(directed=False)
    for i in range(0, 40):
        g2.add_vertex()

    g2.add_edge(0, 4)
    g2.add_edge(0, 6)
    g2.add_edge(0, 9)
    g2.add_edge(1, 5)
    g2.add_edge(1, 7)
    g2.add_edge(1, 9)
    g2.add_edge(2, 4)
    g2.add_edge(2, 7)
    g2.add_edge(2, 8)
    g2.add_edge(3, 5)
    g2.add_edge(3, 6)
    g2.add_edge(3, 8)
    g2.add_edge(4, 20)
    g2.add_edge(5, 21)
    g2.add_edge(6, 11)
    g2.add_edge(7, 10)
    g2.add_edge(8, 30)
    g2.add_edge(9, 31)
    g2.add_edge(10, 12)
    g2.add_edge(10, 15)
    g2.add_edge(11, 13)
    g2.add_edge(11, 14)
    g2.add_edge(12, 18)
    g2.add_edge(12, 17)
    g2.add_edge(13, 16)
    g2.add_edge(13, 17)
    g2.add_edge(14, 18)
    g2.add_edge(14, 19)
    g2.add_edge(15, 16)
    g2.add_edge(15, 19)
    g2.add_edge(16, 23)
    g2.add_edge(17, 32)
    g2.add_edge(18, 25)
    g2.add_edge(19, 34)
    g2.add_edge(20, 24)
    g2.add_edge(20, 28)
    g2.add_edge(21, 22)
    g2.add_edge(21, 26)
    g2.add_edge(23, 24)
    g2.add_edge(23, 26)
    g2.add_edge(25, 22)
    g2.add_edge(25, 28)
    g2.add_edge(27, 22)
    g2.add_edge(27, 24)
    g2.add_edge(29, 26)
    g2.add_edge(29, 28)
    g2.add_edge(31, 33)
    g2.add_edge(31, 36)
    g2.add_edge(30, 37)
    g2.add_edge(30, 39)
    g2.add_edge(32, 33)
    g2.add_edge(32, 39)
    g2.add_edge(34, 37)
    g2.add_edge(34, 35)
    g2.add_edge(36, 37)
    g2.add_edge(36, 39)
    g2.add_edge(38, 33)
    g2.add_edge(38, 35)
    g2.add_edge(27, 36)
    g2.add_edge(29, 28)
    g2.add_edge(29, 26)
    g2.add_edge(29, 38)

    return (g1, g2)


def create_gaurav_2_graphs():
    g1 = Graph(directed=False)
    for i in range(0, 40):
        g1.add_vertex()

    g1.add_edge(0, 1)
    g1.add_edge(0, 12)
    g1.add_edge(0, 15)
    g1.add_edge(0, 16)
    g1.add_edge(0, 17)
    g1.add_edge(1, 13)
    g1.add_edge(1, 14)
    g1.add_edge(1, 18)
    g1.add_edge(1, 19)
    g1.add_edge(2, 3)
    g1.add_edge(2, 12)
    g1.add_edge(2, 14)
    g1.add_edge(2, 20)
    g1.add_edge(2, 21)
    g1.add_edge(3, 13)
    g1.add_edge(3, 15)
    g1.add_edge(3, 22)
    g1.add_edge(3, 23)
    g1.add_edge(4, 5)
    g1.add_edge(4, 12)
    g1.add_edge(4, 13)
    g1.add_edge(4, 24)
    g1.add_edge(4, 26)
    g1.add_edge(5, 14)
    g1.add_edge(5, 15)
    g1.add_edge(5, 25)
    g1.add_edge(5, 27)
    g1.add_edge(6, 7)
    g1.add_edge(6, 16)
    g1.add_edge(6, 19)
    g1.add_edge(6, 20)
    g1.add_edge(6, 22)
    g1.add_edge(7, 17)
    g1.add_edge(7, 18)
    g1.add_edge(7, 21)
    g1.add_edge(7, 23)
    g1.add_edge(8, 9)
    g1.add_edge(8, 16)
    g1.add_edge(8, 18)
    g1.add_edge(8, 24)
    g1.add_edge(8, 25)
    g1.add_edge(9, 17)
    g1.add_edge(9, 19)
    g1.add_edge(9, 26)
    g1.add_edge(9, 27)
    g1.add_edge(10, 11)
    g1.add_edge(10, 20)
    g1.add_edge(10, 23)
    g1.add_edge(10, 24)
    g1.add_edge(10, 27)
    g1.add_edge(11, 21)
    g1.add_edge(11, 22)
    g1.add_edge(11, 25)
    g1.add_edge(11, 26)

    g2 = Graph(directed=False)
    for i in range(0, 40):
        g2.add_vertex()

    g2.add_edge(0, 1)
    g2.add_edge(0, 13)
    g2.add_edge(0, 15)
    g2.add_edge(0, 16)
    g2.add_edge(0, 17)
    g2.add_edge(1, 12)
    g2.add_edge(1, 14)
    g2.add_edge(1, 18)
    g2.add_edge(1, 19)
    g2.add_edge(2, 3)
    g2.add_edge(2, 13)
    g2.add_edge(2, 14)
    g2.add_edge(2, 20)
    g2.add_edge(2, 21)
    g2.add_edge(3, 12)
    g2.add_edge(3, 15)
    g2.add_edge(3, 22)
    g2.add_edge(3, 23)
    g2.add_edge(4, 5)
    g2.add_edge(4, 14)
    g2.add_edge(4, 15)
    g2.add_edge(4, 24)
    g2.add_edge(4, 26)
    g2.add_edge(5, 12)
    g2.add_edge(5, 13)
    g2.add_edge(5, 25)
    g2.add_edge(5, 27)
    g2.add_edge(6, 7)
    g2.add_edge(6, 16)
    g2.add_edge(6, 19)
    g2.add_edge(6, 20)
    g2.add_edge(6, 22)
    g2.add_edge(7, 17)
    g2.add_edge(7, 18)
    g2.add_edge(7, 21)
    g2.add_edge(7, 23)
    g2.add_edge(8, 9)
    g2.add_edge(8, 16)
    g2.add_edge(8, 18)
    g2.add_edge(8, 24)
    g2.add_edge(8, 25)
    g2.add_edge(9, 17)
    g2.add_edge(9, 19)
    g2.add_edge(9, 26)
    g2.add_edge(9, 27)
    g2.add_edge(10, 11)
    g2.add_edge(10, 20)
    g2.add_edge(10, 23)
    g2.add_edge(10, 24)
    g2.add_edge(10, 27)
    g2.add_edge(11, 21)
    g2.add_edge(11, 22)
    g2.add_edge(11, 25)
    g2.add_edge(11, 26)

    return (g1, g2)


def create_sr_graphs():
    g1 = Graph(directed=False)
    a = g1.add_vertex()
    b = g1.add_vertex()
    c = g1.add_vertex()
    d = g1.add_vertex()
    e = g1.add_vertex()
    f = g1.add_vertex()
    g = g1.add_vertex()
    h = g1.add_vertex()
    i = g1.add_vertex()
    j = g1.add_vertex()
    k = g1.add_vertex()
    l = g1.add_vertex()
    m = g1.add_vertex()
    n = g1.add_vertex()
    o = g1.add_vertex()
    p = g1.add_vertex()

    g1.add_edge(a, p)
    g1.add_edge(a, m)
    g1.add_edge(a, i)
    g1.add_edge(a, e)
    g1.add_edge(a, c)
    g1.add_edge(a, b)

    g1.add_edge(b, p)
    g1.add_edge(b, n)
    g1.add_edge(b, j)
    g1.add_edge(b, f)
    g1.add_edge(b, c)

    g1.add_edge(c, p)
    g1.add_edge(c, o)
    g1.add_edge(c, k)
    g1.add_edge(c, g)

    g1.add_edge(d, p)
    g1.add_edge(d, l)
    g1.add_edge(d, h)
    g1.add_edge(d, g)
    g1.add_edge(d, f)
    g1.add_edge(d, e)

    g1.add_edge(e, m)
    g1.add_edge(e, i)
    g1.add_edge(e, g)
    g1.add_edge(e, f)

    g1.add_edge(f, n)
    g1.add_edge(f, j)
    g1.add_edge(f, g)

    g1.add_edge(g, o)
    g1.add_edge(g, k)

    g1.add_edge(h, p)
    g1.add_edge(h, l)
    g1.add_edge(h, k)
    g1.add_edge(h, j)
    g1.add_edge(h, i)

    g1.add_edge(i, m)
    g1.add_edge(i, k)
    g1.add_edge(i, j)

    g1.add_edge(j, n)
    g1.add_edge(j, k)

    g1.add_edge(k, o)

    g1.add_edge(l, m)
    g1.add_edge(l, n)
    g1.add_edge(l, o)
    g1.add_edge(l, p)

    g1.add_edge(m, n)
    g1.add_edge(m, o)

    g1.add_edge(n, o)

    #####################################################
    g2 = Graph(directed=False)
    aa = g2.add_vertex()
    bb = g2.add_vertex()
    cc = g2.add_vertex()
    dd = g2.add_vertex()
    ee = g2.add_vertex()
    ff = g2.add_vertex()
    gg = g2.add_vertex()
    hh = g2.add_vertex()
    ii = g2.add_vertex()
    jj = g2.add_vertex()
    kk = g2.add_vertex()
    ll = g2.add_vertex()
    mm = g2.add_vertex()
    nn = g2.add_vertex()
    oo = g2.add_vertex()
    pp = g2.add_vertex()

    g2.add_edge(aa, hh)
    g2.add_edge(aa, gg)
    g2.add_edge(aa, pp)
    g2.add_edge(aa, jj)
    g2.add_edge(aa, cc)
    g2.add_edge(aa, bb)

    g2.add_edge(bb, hh)
    g2.add_edge(bb, ii)
    g2.add_edge(bb, kk)
    g2.add_edge(bb, dd)
    g2.add_edge(bb, cc)

    g2.add_edge(cc, jj)
    g2.add_edge(cc, ll)
    g2.add_edge(cc, ee)
    g2.add_edge(cc, dd)

    g2.add_edge(dd, mm)
    g2.add_edge(dd, ff)
    g2.add_edge(dd, ee)
    g2.add_edge(dd, kk)

    g2.add_edge(ee, ll)
    g2.add_edge(ee, nn)
    g2.add_edge(ee, gg)
    g2.add_edge(ee, ff)

    g2.add_edge(ff, mm)
    g2.add_edge(ff, oo)
    g2.add_edge(ff, hh)
    g2.add_edge(ff, gg)

    g2.add_edge(gg, nn)
    g2.add_edge(gg, pp)
    g2.add_edge(gg, hh)

    g2.add_edge(hh, oo)
    g2.add_edge(hh, ii)

    g2.add_edge(ii, oo)
    g2.add_edge(ii, nn)
    g2.add_edge(ii, ll)
    g2.add_edge(ii, kk)

    g2.add_edge(jj, pp)
    g2.add_edge(jj, oo)
    g2.add_edge(jj, mm)
    g2.add_edge(jj, ll)

    g2.add_edge(kk, pp)
    g2.add_edge(kk, nn)
    g2.add_edge(kk, mm)

    g2.add_edge(ll, oo)
    g2.add_edge(ll, nn)

    g2.add_edge(mm, pp)
    g2.add_edge(mm, oo)

    g2.add_edge(nn, pp)

    return (g1, g2)


def create_cycle_pair(k):
    # One large cycle.
    cycle_1 = Graph(directed=False)

    for i in range(0, 2 * k):
        cycle_1.add_vertex()

    for i in range(0, 2 * k):
        cycle_1.add_edge(i, (i + 1) % (2 * k))

    # 2 smaller cycles.
    c_1 = Graph(directed=False)

    for i in range(0, k):
        c_1.add_vertex()

    for i in range(0, k):
        c_1.add_edge(i, (i + 1) % k)

    # Second cycle.
    c_2 = Graph(directed=False)
    for i in range(0, k):
        c_2.add_vertex()

    for i in range(0, k):
        c_2.add_edge(i, (i + 1) % k)

    cycle_2 = graph_union(c_1, c_2)

    return (cycle_1, cycle_2)


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

    # Draw for visual inspection.
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
def compute_k_s_tuple_graph(graphs, k, s, directed=False, self_loops=True):
    tupled_graphs = []
    node_labels_all = []
    edge_labels_all = []

    # Manage atomic types.
    atomic_type = {}
    atomic_counter = 0

    for g in graphs:
        # k-tuple graph.
        k_tuple_graph = Graph(directed=directed)

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

                    # Check if tuple exists, otherwise ignore.
                    if tuple_exists[tuple(n)]:
                        w = tuple_to_node[tuple(n)]

                        # Insert edge, avoid undirected multi-edges.
                        if directed:
                            k_tuple_graph.add_edge(m, w)
                            edge_labels[(m, w)] = i + 1
                        else:
                            if not k_tuple_graph.edge(w, m):
                                k_tuple_graph.add_edge(m, w)
                                edge_labels[(m, w)] = i + 1
                                edge_labels[(w, m)] = i + 1

            # Add self-loops, only once.
            if self_loops:
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


graphs = create_gaurav_graphs(k=5)

print("###")
tupled_graphs, node_labels, edge_labels = compute_k_s_tuple_graph(graphs, 2, 1)
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
