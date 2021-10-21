#include <cstdio>
#include <iostream>
#include "src/AuxiliaryMethods.h"
#include "src/Graph.h"

// This might need to adapted to your specific system.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;
using namespace GraphLibrary;

// Generate sparse adjacency matrix representation of two-tuple graph of graph g.
pair<vector<vector<uint>>, vector<vector<uint>>> generate_local_sparse_am_2_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();

    // Maps node in two-tuple graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;

    Node num_two_tuples = 0;
    // Generate all tuples that induce connected graphs on at most two vertices.
    for (Node i = 0; i < num_nodes; ++i) {
        Nodes neighbors = g.get_neighbours(i);

        for (Node j: neighbors) {
            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;
        }

        node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, i)}});
        two_tuple_to_node.insert({{make_tuple(i, i), num_two_tuples}});

        num_two_tuples++;
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;

    for (Node i = 0; i < num_two_tuples; ++i) {
        // Get nodes of original graph corresponding to two tuple i.
        TwoTuple p = node_to_two_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            unordered_map<TwoTuple, Node>::const_iterator t = two_tuple_to_node.find(make_tuple(v_n, w));

            // Check if tuple exists.
            if (t != two_tuple_to_node.end()) {
                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n: w_neighbors) {
            unordered_map<TwoTuple, Node>::const_iterator t = two_tuple_to_node.find(make_tuple(v, w_n));

            // Check if tuple exists.
            if (t != two_tuple_to_node.end()) {
                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }
    }

    return std::make_pair(nonzero_compenents_1, nonzero_compenents_2);
}

// Generate node labels for two-tuple graph of graph g.
vector<unsigned long> get_node_labels_2_1(const Graph &g, const bool use_labels, const bool use_edge_labels) {
    // Get node and edge labels.
    Labels labels;
    vector<unsigned long> tuple_labels;
    if (use_labels) {
        labels = g.get_labels();
    }

    EdgeLabels edge_labels;
    if (use_edge_labels) {
        edge_labels = g.get_edge_labels();
    }

    size_t num_nodes = g.get_num_nodes();
    // Compute labels of all tuples.
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
                cout << "ERROR: Non edges should not appear." << endl;
                exit(-1);
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


// Generate sparse adjacency matrix representation of graph g.
vector<vector<uint>> generate_local_sparse_am_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();
    vector<vector<uint>> nonzero_compenents;

    for (Node v = 0; v < num_nodes; ++v) {
        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            nonzero_compenents.push_back({{v, v_n}});
        }
    }

    return nonzero_compenents;
}


// Get node labels of graph g.
vector<unsigned long> get_node_labels_1(const Graph &g) {
    return g.get_labels();
}

// Get edge labels of graph g.
vector<int> get_edge_labels_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();
    EdgeLabels edge_labels = g.get_edge_labels();
    vector<int> labels;

    for (Node v = 0; v < num_nodes; ++v) {
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            labels.push_back(edge_labels.find(std::make_tuple(v, v_n))->second);
        }
    }

    return labels;
}


// Get all sparse adjacency matrix representations of two-tuple graphs in graph database.
vector<pair<vector<vector<uint>>, vector<vector<uint>>>>
get_all_matrices_2_1(string name, const std::vector<int> &indices) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    GraphDatabase gdb_new;
    for (int i: indices) {
        gdb_new.push_back(gdb[i]);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;
    for (auto &g: gdb_new) {
        matrices.push_back(generate_local_sparse_am_2_1(g));
    }

    return matrices;
}


// Get all node labels of two-tuple graphs in graph database.
vector<vector<unsigned long>> get_all_node_labels_2_1(string name, const bool use_node_labels, const bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);
    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb) {
        vector<unsigned long> colors = get_node_labels_2_1(g, use_node_labels, use_edge_labels);
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

    cout << "Number of different labels: " << m_num_labels << endl;
    return node_labels;
}


// Get all sparse adjacency matrix representations of graphs in graph database.
vector<vector<vector<uint>>> get_all_matrices_1(string name, const std::vector<int> &indices) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    GraphDatabase gdb_new;

    for (int i: indices) {
        gdb_new.push_back(gdb[i]);
    }

    vector<vector<vector<uint>>> matrices;
    for (auto &g: gdb_new) {
        matrices.push_back(generate_local_sparse_am_1(g));
    }

    return matrices;
}


// Get all node labels of graphs in graph database.
vector<vector<unsigned long>> get_all_node_labels_1(string name, const bool use_node_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb) {
        vector<unsigned long> colors = get_node_labels_1(g);
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

    cout << "Number of different labels: " << m_num_labels << endl;
    return node_labels;
}


// Get all edge labels of graphs in graph database.
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
                m_label_to_index.insert({{c, m_num_labels}});
                new_color.push_back(m_num_labels);
                m_num_labels++;
            } else {
                new_color.push_back(it->second);
            }
        }
        edge_labels.push_back(new_color);
    }

    cout << "Number of different labels: " << m_num_labels << endl;
    return edge_labels;
}


vector<vector<unsigned long>>
get_all_node_labels_2_1_ZINC(const bool use_node_labels, const bool use_edge_labels,
                             const std::vector<int> &indices_train,
                             const std::vector<int> &indices_val, const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);
    GraphDatabase gdb_new_train;
    for (auto i: indices_train) {
        gdb_new_train.push_back(gdb_1[int(i)]);
    }

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);
    GraphDatabase gdb_new_val;
    for (auto i: indices_val) {
        gdb_new_val.push_back(gdb_2[int(i)]);
    }

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_test;
    for (auto i: indices_test) {
        gdb_new_test.push_back(gdb_3[int(i)]);
    }
    
    vector<vector<unsigned long>> node_labels;
    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_train) {
        vector<unsigned long> colors = get_node_labels_2_1(g, use_node_labels, use_edge_labels);
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

    for (auto &g: gdb_new_val) {
        vector<unsigned long> colors = get_node_labels_2_1(g, use_node_labels, use_edge_labels);
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

    for (auto &g: gdb_new_test) {
        vector<unsigned long> colors = get_node_labels_2_1(g, use_node_labels, use_edge_labels);
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

    cout << "Number of different labels: " << m_num_labels << endl;
    return node_labels;
}

vector<vector<unsigned long>> get_all_node_labels_ZINC_1(const bool use_node_labels, const bool use_edge_labels,
                                                         const std::vector<int> &indices_train,
                                                         const std::vector<int> &indices_val,
                                                         const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_train;
    for (auto i: indices_train) {
        gdb_new_train.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_train.size() << endl;
    
    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_val;
    for (auto i: indices_val) {
        gdb_new_val.push_back(gdb_2[int(i)]);
    }

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_test;
    for (auto i: indices_test) {
        gdb_new_test.push_back(gdb_3[int(i)]);
    }
    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_train) {
        vector<unsigned long> colors = get_node_labels_1(g);
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

    for (auto &g: gdb_new_val) {
        vector<unsigned long> colors = get_node_labels_1(g);
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

    for (auto &g: gdb_new_test) {
        vector<unsigned long> colors = get_node_labels_1(g);
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

    cout << "Number of different labels: " << m_num_labels << endl;
    return node_labels;

}

vector<vector<int>> get_all_edge_labels_ZINC_1(const bool use_node_labels, const bool use_edge_labels,
                                               const std::vector<int> &indices_train,
                                               const std::vector<int> &indices_val,
                                               const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_train;
    for (auto i: indices_train) {
        gdb_new_train.push_back(gdb_1[int(i)]);
    }

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_new_val;
    for (auto i: indices_val) {
        gdb_new_val.push_back(gdb_2[int(i)]);
    }

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
    GraphDatabase gdb_new_test;
    for (auto i: indices_test) {
        gdb_new_test.push_back(gdb_3[int(i)]);
    }

    vector<vector<int>> edge_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_train) {
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

    for (auto &g: gdb_new_val) {
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

    for (auto &g: gdb_new_test) {
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

    cout << "Number of different labels: " << m_num_labels << endl;
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

PYBIND11_MODULE(preprocessing, m
) {
m.def("get_all_matrices_2_1", &get_all_matrices_2_1);
m.def("get_all_matrices_1", &get_all_matrices_1);

m.def("get_all_node_labels_2_1", &get_all_node_labels_2_1);
m.def("get_all_node_labels_1", &get_all_node_labels_1);
m.def("get_all_edge_labels_1", &get_all_edge_labels_1);

m.def("get_all_node_labels_ZINC_2_1", &get_all_node_labels_ZINC_2_1);
m.def("get_all_node_labels_ZINC_1", &get_all_node_labels_ZINC_1);

m.def("get_all_edge_labels_ZINC_1", &get_all_edge_labels_ZINC_1);

m.def("read_targets", &read_targets);
}