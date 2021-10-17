from graph_tool.all import *

from itertools import chain, combinations

import graph_tool as gt
import networkx as nx
from graph_tool.all import *


def CFI(k):
    K = nx.complete_graph(k + 1)

    ## graph 1

    G = nx.Graph()

    for e in K.edges:
        G.add_node((e, 0))
        G.add_node((e, 1))
        G.add_edge((e, 1), (e, 0))

    for u in K:
        for S in subsetslist(0, K, u):
            G.add_node((u, S))
            for e in incidentedges(K, u):
                G.add_edge((u, S), (e, int(e in S)))

    ## graph 2

    H = nx.Graph()

    for e in K.edges:
        H.add_node((e, 0))
        H.add_node((e, 1))
        H.add_edge((e, 1), (e, 0))

    for u in K:
        parity = int(u == 0)  ## vertex 0 in K, the "odd" one out
        for S in subsetslist(parity, K, u):
            H.add_node((u, S))
            for e in incidentedges(K, u):
                H.add_edge((u, S), (e, int(e in S)))

    G = nx.convert_node_labels_to_integers(G)
    H = nx.convert_node_labels_to_integers(H)

    return (G, H)


## list of edges incident to a vertex,
## where each edge (i,j) satisfies i < j

def incidentedges(K, u):
    return [tuple(sorted(e)) for e in K.edges(u)]


## generate all edge subsets of odd/even cardinality
## set parameter "parity" 0/1 for odd/even sets resp.

def subsetslist(parity, K, u):
    oddsets = set()
    evensets = set()
    for s in list(powerset(incidentedges(K, u))):
        if (len(s) % 2 == 0):
            evensets.add(frozenset(s))
        else:
            oddsets.add(frozenset(s))
    if parity == 0:
        return evensets
    else:
        return oddsets


## generate all subsets of a set
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def networkx_to_gt(nxG):
    gtG = gt.Graph(directed=nxG.is_directed())

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later
    for node in nxG.nodes():
        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

    # Add the edges
    for src, dst in nxG.edges():
        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

    # Done, finally!
    return gtG


def create_gaurav_graphs(k):
    g_1, g_2 = CFI(k)

    g_1 = networkx_to_gt(g_1)
    g_2 = networkx_to_gt(g_2)

    g_1.vp.node_color = g_1.new_vertex_property("int")
    for v in g_1.vertices():
        g_1.vp.node_color[v] = 1

    g_1.ep.edge_color = g_1.new_edge_property("int")
    for e in g_1.edges():
        g_1.ep.edge_color[e] = 2

    g_2.vp.node_color = g_2.new_vertex_property("int")
    for v in g_2.vertices():
        g_2.vp.node_color[v] = 1

    g_2.ep.edge_color = g_2.new_edge_property("int")
    for e in g_2.edges():
        g_2.ep.edge_color[e] = 2

    return (g_1, g_2)


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


# Create cycle counter examples.
def create_pair_four():
    g_1 = Graph(directed=False)
    g_1.add_vertex()
    g_1.add_vertex()
    g_1.add_vertex()
    g_1.add_vertex()
    g_1.add_vertex()
    g_1.add_vertex()
    g_1.add_vertex()
    g_1.add_vertex()

    g_1.add_edge(0, 1)
    g_1.add_edge(1, 2)
    g_1.add_edge(2, 3)
    g_1.add_edge(3, 0)
    g_1.add_edge(3, 4)
    g_1.add_edge(4, 5)
    g_1.add_edge(5, 6)
    g_1.add_edge(6, 7)
    g_1.add_edge(7, 4)

    g_2 = Graph(directed=False)
    g_2.add_vertex()
    g_2.add_vertex()
    g_2.add_vertex()
    g_2.add_vertex()
    g_2.add_vertex()
    g_2.add_vertex()
    g_2.add_vertex()
    g_2.add_vertex()

    g_2.add_edge(0, 1)
    g_2.add_edge(1, 2)
    g_2.add_edge(2, 3)
    g_2.add_edge(3, 4)
    g_2.add_edge(4, 0)
    g_2.add_edge(3, 5)
    g_2.add_edge(5, 6)
    g_2.add_edge(6, 7)
    g_2.add_edge(7, 2)

    return [g_1, g_2]
