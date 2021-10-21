/**/
#include <cstdio>
#include "src/AuxiliaryMethods.h"
#include "src/Graph.h"
#include <iostream>


#ifdef __linux__
//#include <pybind11/pybind11.h>
//#include <pybind11/eigen.h>
//#include <pybind11/stl.h>
#include </home/morrchri/.local/include/python3.8/pybind11/pybind11.h>
#include </home/morrchri/.local/include/python3.8/pybind11/eigen.h>
#include </home/morrchri/.local/include/python3.8/pybind11/stl.h>
#else
// MacOS.
#include </usr/local/include/pybind11/pybind11.h>
#include </usr/local/include/pybind11/stl.h>
#include </usr/local/include/pybind11/eigen.h>

#endif


namespace py = pybind11;
using namespace std;
using namespace GraphLibrary;

using namespace std;

pair<vector<vector<uint>>, vector<vector<uint>>>
generate_local_sparse_am(const Graph &g, const bool use_labels, const bool use_edge_labels) {
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels) {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels) {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i) {
        Nodes neighbors = g.get_neighbours(i);

        for (Node j: neighbors) {

            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

        }

        node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, i)}});
        two_tuple_to_node.insert({{make_tuple(i, i), num_two_tuples}});

        num_two_tuples++;

    }

    vector<vector<uint >> nonzero_compenents_1;
    vector<vector<uint >> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i) {
        // Get nodes of original graph corresponding to two tuple i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            unordered_map<TwoTuple, Node>::const_iterator t = two_tuple_to_node.find(make_tuple(v_n, w));

            if (t != two_tuple_to_node.end()) {
                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n: w_neighbors) {
            unordered_map<TwoTuple, Node>::const_iterator t = two_tuple_to_node.find(make_tuple(v, w_n));

            if (t != two_tuple_to_node.end()) {
                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }
    }

    return std::make_pair(nonzero_compenents_1, nonzero_compenents_2);
}


vector<vector<uint>> generate_local_sparse_am_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();

    vector<vector<uint >> nonzero_compenents;

    for (Node v = 0; v < num_nodes; ++v) {
        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            nonzero_compenents.push_back({{v, v_n}});
        }
    }

    return nonzero_compenents;
}


pair<vector<int>, vector<int>> get_edge_labels(const Graph &g, const bool use_labels, const bool use_edge_labels) {
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.
    Graph two_tuple_graph(false);

    // Maps node in two set graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;
    // Manages edges labels.
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Create a node for each two set.
    Labels labels;
    Labels tuple_labels;
    if (use_labels) {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels) {
        edge_labels = g.get_edge_labels();
    }

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i) {
        for (Node j = 0; j < num_nodes; ++j) {

            two_tuple_graph.add_node();

            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;

            Label c_i = 1;
            Label c_j = 2;
            if (use_labels) {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j)) {
                if (use_edge_labels) {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                } else {
                    c = 3;
                }
            } else if (i == j) {
                c = 1;
            } else {
                c = 2;
            }

            Label new_color = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }
    }


    vector<int> edge_labelsn_1;
    vector<int> edge_labelsn_2;
    for (Node i = 0; i < num_two_tuples; ++i) {
        // Get nodes of orginal graph corresponding to two set i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i) {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v_i, w));

            // Local vs. global edge.
            if (g.has_edge(v, v_i)) {
                edge_labelsn_1.push_back(0);

            } else {
                edge_labelsn_1.push_back(1);

            }

            two_tuple_graph.add_edge(i, t->second);
        }
        // Exchange second node.
        // Iterate over nodes.
        for (Node v_i = 0; v_i < num_nodes; ++v_i) {
            unordered_map<TwoTuple, Node>::const_iterator t;
            t = two_tuple_to_node.find(make_tuple(v, v_i));

            // Local vs. global edge.
            if (g.has_edge(w, v_i)) {
                edge_labelsn_2.push_back(0);

            } else {
                edge_labelsn_2.push_back(1);
            }

            two_tuple_graph.add_edge(i, t->second);
        }
    }

    return std::make_pair(edge_labelsn_1, edge_labelsn_2);
}

vector<unsigned long> get_node_labels(const Graph &g, const bool use_labels, const bool use_edge_labels) {
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels) {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels) {
        edge_labels = g.get_edge_labels();
    }

    for (Node i = 0; i < num_nodes; ++i) {
            Nodes neighbors = g.get_neighbours(i);

        for (Node j: neighbors) {
            Label c_i = 1;
            Label c_j = 2;
            if (use_labels) {
                c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
                c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);
            }

            Label c;
            if (g.has_edge(i, j)) {
                if (use_edge_labels) {
                    auto s = edge_labels.find(make_tuple(i, j));
                    c = AuxiliaryMethods::pairing(3, s->second);
                } else {
                    c = 3;
                }
            } else if (i == j) {
                c = 1;
            } else {
                c = 2;
            }

            Label new_color = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), c);
            tuple_labels.push_back(new_color);
        }

        Label c_i = 1;
        Label c_j = 2;
        if (use_labels) {
            c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
            c_j = AuxiliaryMethods::pairing(labels[i] + 1, c_j);
        }

        Label new_color = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), 1);
        tuple_labels.push_back(new_color);
    }

    return tuple_labels;
}


vector<unsigned long> get_node_labels_1(const Graph &g, const bool use_labels) {
    size_t num_nodes = g.get_num_nodes();

    // Create a node for each two set.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels) {
        tuple_labels = g.get_labels();
    }

    return tuple_labels;
}


vector<int> get_edge_labels_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();
    EdgeLabels edge_labels = g.get_edge_labels();

    vector<int> labels;

    for (Node v = 0; v < num_nodes; ++v) {
        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            labels.push_back(edge_labels.find(std::make_tuple(v, v_n))->second);
        }
    }

    return labels;
}


vector<pair<vector<vector<uint>>, vector<vector<uint>>>> get_all_matrices(string name, const std::vector<int> &indices) {

    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    GraphDatabase gdb_new;

    for (auto i: indices) {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;

    for (auto &g: gdb_new) {
        matrices.push_back(generate_local_sparse_am(g, false, false));
    }

    return matrices;
}

vector<vector<vector<uint>>> get_all_matrices_1(string name, const std::vector<int> &indices) {

    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    GraphDatabase gdb_new;

    for (auto i: indices) {
        gdb_new.push_back(gdb[int(i)]);
    }

    vector<vector<vector<uint>>> matrices;

    for (auto &g: gdb_new) {
        matrices.push_back(generate_local_sparse_am_1(g));
    }

    return matrices;
}


vector<vector<int>> get_all_edge_labels_1(string name) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    vector<vector<int>> edge_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb) {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{ c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;

    return edge_labels;
}


vector<vector<unsigned long>> get_all_node_labels(string name, const bool use_node_labels, const bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb) {
        vector<unsigned long> colors = get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}


vector<vector<unsigned long>> get_all_node_labels_1(string name, const bool use_node_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin()+ 0);

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb) {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}


vector<vector<unsigned long>>
get_all_node_labels_ZINC(const bool use_node_labels, const bool use_edge_labels, const std::vector<int> &indices_train,
                         const std::vector<int> &indices_val, const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i: indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;


    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i: indices_val) {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i: indices_test) {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<unsigned long> colors = get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g: gdb_new_2) {
        vector<unsigned long> colors = get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g: gdb_new_3) {
        vector<unsigned long> colors = get_node_labels(g, use_node_labels, use_edge_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_ZINC_1(const bool use_node_labels, const bool use_edge_labels,
                                                         const std::vector<int> &indices_train,
                                                         const std::vector<int> &indices_val,
                                                         const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i: indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;


    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i: indices_val) {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i: indices_test) {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g: gdb_new_2) {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    for (auto &g: gdb_new_3) {
        vector<unsigned long> colors = get_node_labels_1(g, use_node_labels);
        vector<unsigned long> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        node_labels.push_back(new_color);
    }

    cout << m_num_labels << endl;
    return node_labels;

}


vector<vector<int>> get_all_edge_labels_ZINC_1(const bool use_node_labels, const bool use_edge_labels,
                                               const std::vector<int> &indices_train,
                                               const std::vector<int> &indices_val,
                                               const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i: indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;
    cout << "$$$" << endl;


    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_2;
    for (auto i: indices_val) {
        gdb_new_2.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_2.size() << endl;
    cout << "$$$" << endl;

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_3;
    for (auto i: indices_test) {
        gdb_new_3.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_3.size() << endl;
    cout << "$$$" << endl;


    vector<vector<int>> edge_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }


    for (auto &g: gdb_new_2) {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }

    for (auto &g: gdb_new_3) {
        vector<int> colors = get_edge_labels_1(g);
        vector<int> new_color;

        for (auto &c: colors) {
            const auto it(m_label_to_index.find(c));
            if (it == m_label_to_index.end()) {
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);

                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }

        edge_labels.push_back(new_color);
    }


    cout << m_num_labels << endl;
    return edge_labels;

}


vector<int> read_classes(string data_set_name) {
    return AuxiliaryMethods::read_classes(data_set_name);
}

vector<float> read_targets(string data_set_name, const std::vector<int> &indices) {
    vector<float> targets = AuxiliaryMethods::read_targets(data_set_name);

    vector<float> new_targets;
    for (auto i: indices) {
        new_targets.push_back(targets[i]);
    }

    return new_targets;
}

PYBIND11_MODULE(preprocessing, m) {
    m.def("get_all_matrices", &get_all_matrices);
    m.def("get_all_matrices_1", &get_all_matrices_1);

    m.def("get_all_node_labels", &get_all_node_labels);
    m.def("get_all_node_labels_1", &get_all_node_labels_1);
    m.def("get_all_edge_labels_1", &get_all_edge_labels_1);

    m.def("get_all_node_labels_ZINC", &get_all_node_labels_ZINC);
    m.def("get_all_node_labels_ZINC_1", &get_all_node_labels_ZINC_1);

    m.def("get_all_edge_labels_ZINC_1", &get_all_edge_labels_ZINC_1);

    m.def("read_targets", &read_targets);
}