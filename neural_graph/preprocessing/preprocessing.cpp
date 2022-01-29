#include <cstdio>
#include <iostream>
#include "src/AuxiliaryMethods.h"
#include "src/Graph.h"

// This might need to adapted to your specific system.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
//#include </home/XXX/.local/include/python3.8/pybind11/pybind11.h>
//#include </home/XXX/.local/include/python3.8/pybind11/eigen.h>
//#include </home/XXX/.local/include/python3.8/pybind11/stl.h>

namespace py = pybind11;
using namespace std;
using namespace GraphLibrary;

tuple <Attributes, Attributes, Attributes, Attributes> get_attributes_3_2(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.

    // Get continious node and edge information.
    Attributes attributes;
    attributes = g.get_attributes();

    EdgeAttributes edge_attributes;
    edge_attributes = g.get_edge_attributes();

    Attributes first;
    Attributes second;
    Attributes third;
    Attributes fourth;

    // Maps node in two set graph to correponding two set.
    unordered_map<Node, ThreeTuple> node_to_three_tuple;
    // Inverse of the above map.
    unordered_map<ThreeTuple, Node> three_tuple_to_node;
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Generate 1-multiset.
    vector<Node> one_multiset;
    for (Node i = 0; i < num_nodes; ++i) {
        one_multiset.push_back(i);
    }

    vector<vector<Node>> two_multiset;

    // Avoid duplicates.
    unordered_map<vector<Node>, bool, VectorHasher> two_multiset_exits;

    for (Node v: one_multiset) {
        for (Node w: one_multiset) {
            vector<Node> new_multiset = {v};
            new_multiset.push_back(w);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = two_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == two_multiset_exits.end()) {
                two_multiset_exits.insert({{new_multiset, true}});

                two_multiset.push_back(new_multiset);
            }
        }
    }

    vector<vector<Node>> three_multiset;
    unordered_map<vector<Node>, bool, VectorHasher> three_multiset_exits;
    for (vector<Node> ms: two_multiset) {
        for (Node v: ms) {
            for (Node w: g.get_neighbours(v)) {
                vector<Node> new_multiset = {ms[0], ms[1]};
                new_multiset.push_back(w);

                std::sort(new_multiset.begin(), new_multiset.end());

                auto t = three_multiset_exits.find(new_multiset);

                // Check if not already exists.
                if (t == three_multiset_exits.end()) {
                    three_multiset_exits.insert({{new_multiset, true}});

                    three_multiset.push_back(new_multiset);
                }
            }

            vector<Node> new_multiset = {ms[0], ms[1]};
            new_multiset.push_back(v);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = three_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == three_multiset_exits.end()) {
                three_multiset_exits.insert({{new_multiset, true}});

                three_multiset.push_back(new_multiset);
            }
        }
    }


    vector<vector<Node>> three_tuples;
    for (vector<Node> ms: three_multiset) {
        three_tuples.push_back({{ms[0], ms[1], ms[2]}});
        three_tuples.push_back({{ms[0], ms[2], ms[1]}});

        three_tuples.push_back({{ms[1], ms[2], ms[0]}});
        three_tuples.push_back({{ms[1], ms[0], ms[2]}});

        three_tuples.push_back({{ms[2], ms[1], ms[0]}});
        three_tuples.push_back({{ms[2], ms[0], ms[1]}});
    }


    size_t num_three_tuples = 0;
    for (vector<Node> tuple: three_tuples) {
        Node i = tuple[0];
        Node j = tuple[1];
        Node k = tuple[2];


        node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
        three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
        num_three_tuples++;

        Attribute attr_i = attributes[i];
        Attribute attr_j = attributes[j];
        Attribute attr_k = attributes[k];

        Attribute e_attr_ij;
        Attribute e_attr_ik;
        Attribute e_attr_jk;

        if (g.has_edge(i, j)) {
            e_attr_ij = edge_attributes.find(std::make_pair(i, j))->second;
        } else {
            e_attr_ij = vector<float>({{0, 0, 0, 0}});
        }

        if (g.has_edge(i, k)) {
            e_attr_ik = edge_attributes.find(std::make_pair(i, k))->second;
        } else {
            e_attr_ik = vector<float>({{0, 0, 0, 0}});
        }

        if (g.has_edge(j, k)) {
            e_attr_jk = edge_attributes.find(std::make_pair(j, k))->second;
        } else {
            e_attr_jk = vector<float>({{0, 0, 0, 0}});
        }

        Attribute e_all;
        e_all.insert(e_all.end(), e_attr_ij.begin(), e_attr_ij.end());
        e_all.insert(e_all.end(), e_attr_ik.begin(), e_attr_ik.end());
        e_all.insert(e_all.end(), e_attr_jk.begin(), e_attr_jk.end());

        first.push_back(attr_i);
        second.push_back(attr_j);
        third.push_back(attr_k);
        fourth.push_back(e_all);
    }

    return std::make_tuple(first, second, third, fourth);
}


tuple <Attributes, Attributes, Attributes, Attributes> get_attributes_3_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.

    // Get continious node and edge information.
    Attributes attributes;
    attributes = g.get_attributes();

    EdgeAttributes edge_attributes;
    edge_attributes = g.get_edge_attributes();

    Attributes first;
    Attributes second;
    Attributes third;
    Attributes fourth;

    // Maps node in two set graph to correponding two set.
    unordered_map<Node, ThreeTuple> node_to_three_tuple;
    // Inverse of the above map.
    unordered_map<ThreeTuple, Node> three_tuple_to_node;
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;

    // Generate 1-multiset.
    vector<Node> one_multiset;
    for (Node i = 0; i < num_nodes; ++i) {
        one_multiset.push_back(i);
    }

    vector<vector<Node>> two_multiset;

    // Avoid duplicates.
    unordered_map<vector<Node>, bool, VectorHasher> two_multiset_exits;

    for (Node v: one_multiset) {
        for (Node w: g.get_neighbours(v)) {
            vector<Node> new_multiset = {v};
            new_multiset.push_back(w);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = two_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == two_multiset_exits.end()) {
                two_multiset_exits.insert({{new_multiset, true}});

                two_multiset.push_back(new_multiset);
            }
        }

        vector<Node> new_multiset = {v};
        new_multiset.push_back(v);

        std::sort(new_multiset.begin(), new_multiset.end());

        auto t = two_multiset_exits.find(new_multiset);

        // Check if not already exists.
        if (t == two_multiset_exits.end()) {
            two_multiset_exits.insert({{new_multiset, true}});

            two_multiset.push_back(new_multiset);
        }
    }

    vector<vector<Node>> three_multiset;
    unordered_map<vector<Node>, bool, VectorHasher> three_multiset_exits;
    for (vector<Node> ms: two_multiset) {
        for (Node v: ms) {
            for (Node w: g.get_neighbours(v)) {
                vector<Node> new_multiset = {ms[0], ms[1]};
                new_multiset.push_back(w);

                std::sort(new_multiset.begin(), new_multiset.end());

                auto t = three_multiset_exits.find(new_multiset);

                // Check if not already exists.
                if (t == three_multiset_exits.end()) {
                    three_multiset_exits.insert({{new_multiset, true}});

                    three_multiset.push_back(new_multiset);
                }
            }

            vector<Node> new_multiset = {ms[0], ms[1]};
            new_multiset.push_back(v);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = three_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == three_multiset_exits.end()) {
                three_multiset_exits.insert({{new_multiset, true}});

                three_multiset.push_back(new_multiset);
            }
        }
    }

    vector<vector<Node>> three_tuples;
    for (vector<Node> ms: three_multiset) {
        three_tuples.push_back({{ms[0], ms[1], ms[2]}});
        three_tuples.push_back({{ms[0], ms[2], ms[1]}});

        three_tuples.push_back({{ms[1], ms[2], ms[0]}});
        three_tuples.push_back({{ms[1], ms[0], ms[2]}});

        three_tuples.push_back({{ms[2], ms[1], ms[0]}});
        three_tuples.push_back({{ms[2], ms[0], ms[1]}});
    }

    size_t num_three_tuples = 0;
    for (vector<Node> tuple: three_tuples) {
        Node i = tuple[0];
        Node j = tuple[1];
        Node k = tuple[2];

        node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
        three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
        num_three_tuples++;

        Attribute attr_i = attributes[i];
        Attribute attr_j = attributes[j];
        Attribute attr_k = attributes[k];

        Attribute e_attr_ij;
        Attribute e_attr_ik;
        Attribute e_attr_jk;

        if (g.has_edge(i, j)) {
            e_attr_ij = edge_attributes.find(std::make_pair(i, j))->second;
        } else {
            e_attr_ij = vector<float>({{0, 0, 0, 0}});
        }

        if (g.has_edge(i, k)) {
            e_attr_ik = edge_attributes.find(std::make_pair(i, k))->second;
        } else {
            e_attr_ik = vector<float>({{0, 0, 0, 0}});
        }

        if (g.has_edge(j, k)) {
            e_attr_jk = edge_attributes.find(std::make_pair(j, k))->second;
        } else {
            e_attr_jk = vector<float>({{0, 0, 0, 0}});
        }

        Attribute e_all;
        e_all.insert(e_all.end(), e_attr_ij.begin(), e_attr_ij.end());
        e_all.insert(e_all.end(), e_attr_ik.begin(), e_attr_ik.end());
        e_all.insert(e_all.end(), e_attr_jk.begin(), e_attr_jk.end());

        first.push_back(attr_i);
        second.push_back(attr_j);
        third.push_back(attr_k);
        fourth.push_back(e_all);
    }

    return std::make_tuple(first, second, third, fourth);
}


vector <tuple<Attributes, Attributes, Attributes, Attributes>> get_all_attributes_3_2(string name, const std::vector<int> &indices) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    vector <tuple<Attributes, Attributes, Attributes, Attributes>> attributes;

    GraphDatabase gdb_new_1;
    for (auto i : indices) {
        gdb_new_1.push_back(gdb[int(i)]);
    }
    cout << gdb_new_1.size() << endl;


    uint i = 1;
    for (auto &g: gdb_new_1) {
       attributes.push_back(get_attributes_3_2(g));
       cout << i << endl;

       i++;
    }

    return attributes;
}


vector <tuple<Attributes, Attributes, Attributes, Attributes>> get_all_attributes_3_1(string name) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    vector <tuple<Attributes, Attributes, Attributes, Attributes>> attributes;

    uint i = 1;
    for (auto &g: gdb) {
       attributes.push_back(get_attributes_3_1(g));
       cout << i << endl;

       i++;
    }

    return attributes;
}


tuple <Attributes, Attributes, Attributes> get_attributes_2_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();

    // Get continious node and edge information.
    Attributes attributes;
    attributes = g.get_attributes();

    EdgeAttributes edge_attributes;
    edge_attributes = g.get_edge_attributes();

    Attributes first;
    Attributes second;
    Attributes third;

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i) {
        for (Node j: g.get_neighbours(i)) {
            // Map each pair to node in two set graph and also inverse.

            Attribute attr_i = attributes[i];
            Attribute attr_j = attributes[j];

            Attribute e_attr_ij;
            if (g.has_edge(i, j)) {
                e_attr_ij = edge_attributes.find(std::make_pair(i, j))->second;
                //cout << attr_i.size() << " " << e_attr_ij.size() << endl;
            } else {
                e_attr_ij = vector<float>({{0, 0, 0, 0}});
            }

            first.push_back(attr_i);
            second.push_back(attr_j);
            third.push_back(e_attr_ij);
        }

        Attribute attr_i = attributes[i];
        Attribute attr_j = attributes[i];

        Attribute e_attr_ij;
        e_attr_ij = vector<float>({{0, 0, 0, 0}});

        first.push_back(attr_i);
        second.push_back(attr_j);
        third.push_back(e_attr_ij);
    }

    return std::make_tuple(first, second, third);
}



tuple <Attributes, Attributes, Attributes> get_attributes_2_2(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();

    // Get continious node and edge information.
    Attributes attributes;
    attributes = g.get_attributes();

    EdgeAttributes edge_attributes;
    edge_attributes = g.get_edge_attributes();

    Attributes first;
    Attributes second;
    Attributes third;

    Node num_two_tuples = 0;
    for (Node i = 0; i < num_nodes; ++i) {
        for (Node j = 0; j < num_nodes; ++j) {
            // Map each pair to node in two set graph and also inverse.

            Attribute attr_i = attributes[i];
            Attribute attr_j = attributes[j];

            Attribute e_attr_ij;
            if (g.has_edge(i, j)) {
                e_attr_ij = edge_attributes.find(std::make_pair(i, j))->second;
                //cout << attr_i.size() << " " << e_attr_ij.size() << endl;
            } else {
                e_attr_ij = vector<float>({{0, 0, 0, 0}});
            }

            first.push_back(attr_i);
            second.push_back(attr_j);
            third.push_back(e_attr_ij);
        }
    }

    return std::make_tuple(first, second, third);
}


vector <tuple<Attributes, Attributes, Attributes>> get_all_attributes_2_1(string name) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    vector <tuple<Attributes, Attributes, Attributes>> attributes;

    uint i = 1;
    for (auto &g: gdb) {
       attributes.push_back(get_attributes_2_1(g));
       cout << i << endl;

       i++;
    }

    return attributes;
}


vector <tuple<Attributes, Attributes, Attributes>> get_all_attributes_2_2(string name) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    vector <tuple<Attributes, Attributes, Attributes>> attributes;

    uint i = 1;
    for (auto &g: gdb) {
       attributes.push_back(get_attributes_2_2(g));
       cout << i << endl;

       i++;
    }

    return attributes;
}


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
        // Get nodes of original graph making up the two tuple i.
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

// Generate sparse adjacency matrix representation of two-tuple graph of graph g.
pair<vector<vector<uint>>, vector<vector<uint>>> generate_local_sparse_am_2_2(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();

    // Maps node in two-tuple graph to corresponding two tuple.
    unordered_map<Node, TwoTuple> node_to_two_tuple;
    // Inverse of the above map.
    unordered_map<TwoTuple, Node> two_tuple_to_node;

    Node num_two_tuples = 0;
    // Generate all tuples that induce connected graphs on at most two vertices.
    for (Node i = 0; i < num_nodes; ++i) {
        for (Node j = 0; j < num_nodes; ++j) {
            // Map each pair to node in two set graph and also inverse.
            node_to_two_tuple.insert({{num_two_tuples, make_tuple(i, j)}});
            two_tuple_to_node.insert({{make_tuple(i, j), num_two_tuples}});
            num_two_tuples++;
        }
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


// Generate sparse adjacency matrix representation of three-tuple graph of graph g.
tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>> generate_local_sparse_am_3_2(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();

    // Maps node in two set graph to correponding two set.
    unordered_map<Node, ThreeTuple> node_to_three_tuple;
    // Inverse of the above map.
    unordered_map<ThreeTuple, Node> three_tuple_to_node;
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;


    // Generate 1-multiset.
    vector<Node> one_multiset;
    for (Node i = 0; i < num_nodes; ++i) {
        one_multiset.push_back(i);
    }

    vector<vector<Node>> two_multiset;

    // Avoid duplicates.
    unordered_map<vector<Node>, bool, VectorHasher> two_multiset_exits;

    for (Node v: one_multiset) {
        for (Node w: one_multiset) {
            vector<Node> new_multiset = {v};
            new_multiset.push_back(w);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = two_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == two_multiset_exits.end()) {
                two_multiset_exits.insert({{new_multiset, true}});

                two_multiset.push_back(new_multiset);
            }
        }
    }

    vector<vector<Node>> three_multiset;
    unordered_map<vector<Node>, bool, VectorHasher> three_multiset_exits;
    for (vector<Node> ms: two_multiset) {
        for (Node v: ms) {
            for (Node w: g.get_neighbours(v)) {
                vector<Node> new_multiset = {ms[0], ms[1]};
                new_multiset.push_back(w);

                std::sort(new_multiset.begin(), new_multiset.end());

                auto t = three_multiset_exits.find(new_multiset);

                // Check if not already exists.
                if (t == three_multiset_exits.end()) {
                    three_multiset_exits.insert({{new_multiset, true}});

                    three_multiset.push_back(new_multiset);
                }
            }

            vector<Node> new_multiset = {ms[0], ms[1]};
            new_multiset.push_back(v);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = three_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == three_multiset_exits.end()) {
                three_multiset_exits.insert({{new_multiset, true}});

                three_multiset.push_back(new_multiset);
            }
        }
    }


    vector<vector<Node>> three_tuples;
    for (vector<Node> ms: three_multiset) {
        three_tuples.push_back({{ms[0], ms[1], ms[2]}});
        three_tuples.push_back({{ms[0], ms[2], ms[1]}});

        three_tuples.push_back({{ms[1], ms[2], ms[0]}});
        three_tuples.push_back({{ms[1], ms[0], ms[2]}});

        three_tuples.push_back({{ms[2], ms[1], ms[0]}});
        three_tuples.push_back({{ms[2], ms[0], ms[1]}});
    }

    size_t num_three_tuples = 0;
    for (vector<Node> tuple: three_tuples) {
        Node i = tuple[0];
        Node j = tuple[1];
        Node k = tuple[2];

        node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
        three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
        num_three_tuples++;
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;
    vector<vector<uint>> nonzero_compenents_3;

    for (Node i = 0; i < num_three_tuples; ++i) {
        // Get nodes of original graph corresponding to two tuple i.
        ThreeTuple p = node_to_three_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);
        Node u = std::get<2>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            unordered_map<ThreeTuple, Node>::const_iterator t = three_tuple_to_node.find(make_tuple(v_n, w, u));

            // Check if tuple exists.
            if (t != three_tuple_to_node.end()) {
                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n: w_neighbors) {
            unordered_map<ThreeTuple, Node>::const_iterator t = three_tuple_to_node.find(make_tuple(v, w_n, u));

            // Check if tuple exists.
            if (t != three_tuple_to_node.end()) {
                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }

        // Exchange third node.
        Nodes u_neighbors = g.get_neighbours(u);
        for (Node u_n: u_neighbors) {
            unordered_map<ThreeTuple, Node>::const_iterator t = three_tuple_to_node.find(make_tuple(v, w, u_n));

            // Check if tuple exists.
            if (t != three_tuple_to_node.end()) {
                nonzero_compenents_3.push_back({{i, t->second}});
            }
        }
    }

    return std::make_tuple(nonzero_compenents_1, nonzero_compenents_2, nonzero_compenents_3);
}


// Generate sparse adjacency matrix representation of three-tuple graph of graph g.
tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>> generate_local_sparse_am_3_1(const Graph &g) {
    size_t num_nodes = g.get_num_nodes();

    // Maps node in two set graph to correponding two set.
    unordered_map<Node, ThreeTuple> node_to_three_tuple;
    // Inverse of the above map.
    unordered_map<ThreeTuple, Node> three_tuple_to_node;
    unordered_map<Edge, uint> edge_type;
    // Manages vertex ids
    unordered_map<Edge, uint> vertex_id;
    unordered_map<Edge, uint> local;


    // Generate 1-multiset.
    vector<Node> one_multiset;
    for (Node i = 0; i < num_nodes; ++i) {
        one_multiset.push_back(i);
    }

    vector<vector<Node>> two_multiset;

    // Avoid duplicates.
    unordered_map<vector<Node>, bool, VectorHasher> two_multiset_exits;

    for (Node v: one_multiset) {
        for (Node w: g.get_neighbours(v)) {
            vector<Node> new_multiset = {v};
            new_multiset.push_back(w);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = two_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == two_multiset_exits.end()) {
                two_multiset_exits.insert({{new_multiset, true}});

                two_multiset.push_back(new_multiset);
            }
        }

        vector<Node> new_multiset = {v};
        new_multiset.push_back(v);

        std::sort(new_multiset.begin(), new_multiset.end());

        auto t = two_multiset_exits.find(new_multiset);

        // Check if not already exists.
        if (t == two_multiset_exits.end()) {
            two_multiset_exits.insert({{new_multiset, true}});

            two_multiset.push_back(new_multiset);
        }
    }

    vector<vector<Node>> three_multiset;
    unordered_map<vector<Node>, bool, VectorHasher> three_multiset_exits;
    for (vector<Node> ms: two_multiset) {
        for (Node v: ms) {
            for (Node w: g.get_neighbours(v)) {
                vector<Node> new_multiset = {ms[0], ms[1]};
                new_multiset.push_back(w);

                std::sort(new_multiset.begin(), new_multiset.end());

                auto t = three_multiset_exits.find(new_multiset);

                // Check if not already exists.
                if (t == three_multiset_exits.end()) {
                    three_multiset_exits.insert({{new_multiset, true}});

                    three_multiset.push_back(new_multiset);
                }
            }

            vector<Node> new_multiset = {ms[0], ms[1]};
            new_multiset.push_back(v);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = three_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == three_multiset_exits.end()) {
                three_multiset_exits.insert({{new_multiset, true}});

                three_multiset.push_back(new_multiset);
            }
        }
    }


    vector<vector<Node>> three_tuples;
    for (vector<Node> ms: three_multiset) {
        three_tuples.push_back({{ms[0], ms[1], ms[2]}});
        three_tuples.push_back({{ms[0], ms[2], ms[1]}});

        three_tuples.push_back({{ms[1], ms[2], ms[0]}});
        three_tuples.push_back({{ms[1], ms[0], ms[2]}});

        three_tuples.push_back({{ms[2], ms[1], ms[0]}});
        three_tuples.push_back({{ms[2], ms[0], ms[1]}});
    }

    size_t num_three_tuples = 0;
    for (vector<Node> tuple: three_tuples) {
        Node i = tuple[0];
        Node j = tuple[1];
        Node k = tuple[2];

        node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
        three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
        num_three_tuples++;
    }

    vector<vector<uint>> nonzero_compenents_1;
    vector<vector<uint>> nonzero_compenents_2;
    vector<vector<uint>> nonzero_compenents_3;

    for (Node i = 0; i < num_three_tuples; ++i) {
        // Get nodes of original graph corresponding to two tuple i.
        ThreeTuple p = node_to_three_tuple.find(i)->second;
        Node v = std::get<0>(p);
        Node w = std::get<1>(p);
        Node u = std::get<2>(p);

        // Exchange first node.
        Nodes v_neighbors = g.get_neighbours(v);
        for (Node v_n: v_neighbors) {
            unordered_map<ThreeTuple, Node>::const_iterator t = three_tuple_to_node.find(make_tuple(v_n, w, u));

            // Check if tuple exists.
            if (t != three_tuple_to_node.end()) {
                nonzero_compenents_1.push_back({{i, t->second}});
            }
        }

        // Exchange second node.
        Nodes w_neighbors = g.get_neighbours(w);
        for (Node w_n: w_neighbors) {
            unordered_map<ThreeTuple, Node>::const_iterator t = three_tuple_to_node.find(make_tuple(v, w_n, u));

            // Check if tuple exists.
            if (t != three_tuple_to_node.end()) {
                nonzero_compenents_2.push_back({{i, t->second}});
            }
        }

        // Exchange third node.
        Nodes u_neighbors = g.get_neighbours(u);
        for (Node u_n: u_neighbors) {
            unordered_map<ThreeTuple, Node>::const_iterator t = three_tuple_to_node.find(make_tuple(v, w, u_n));

            // Check if tuple exists.
            if (t != three_tuple_to_node.end()) {
                nonzero_compenents_3.push_back({{i, t->second}});
            }
        }
    }

    return std::make_tuple(nonzero_compenents_1, nonzero_compenents_2, nonzero_compenents_3);
}



// Generate node labels for two-tuple graph of graph g.
vector<unsigned long> get_node_labels_3_2(const Graph &g, const bool use_labels, const bool use_edge_labels) {
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.

    // Maps node in two set graph to correponding two set.
    unordered_map<Node, ThreeTuple> node_to_three_tuple;
    // Inverse of the above map.
    unordered_map<ThreeTuple, Node> three_tuple_to_node;
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


    // Generate 1-multiset.
    vector<Node> one_multiset;
    for (Node i = 0; i < num_nodes; ++i) {
        one_multiset.push_back(i);
    }

    vector<vector<Node>> two_multiset;

    // Avoid duplicates.
    unordered_map<vector<Node>, bool, VectorHasher> two_multiset_exits;

    for (Node v: one_multiset) {
        for (Node w: one_multiset) {
            vector<Node> new_multiset = {v};
            new_multiset.push_back(w);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = two_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == two_multiset_exits.end()) {
                two_multiset_exits.insert({{new_multiset, true}});

                two_multiset.push_back(new_multiset);
            }
        }
    }

    vector<vector<Node>> three_multiset;
    unordered_map<vector<Node>, bool, VectorHasher> three_multiset_exits;
    for (vector<Node> ms: two_multiset) {
        for (Node v: ms) {
            for (Node w: g.get_neighbours(v)) {
                vector<Node> new_multiset = {ms[0], ms[1]};
                new_multiset.push_back(w);

                std::sort(new_multiset.begin(), new_multiset.end());

                auto t = three_multiset_exits.find(new_multiset);

                // Check if not already exists.
                if (t == three_multiset_exits.end()) {
                    three_multiset_exits.insert({{new_multiset, true}});

                    three_multiset.push_back(new_multiset);
                }
            }

            vector<Node> new_multiset = {ms[0], ms[1]};
            new_multiset.push_back(v);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = three_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == three_multiset_exits.end()) {
                three_multiset_exits.insert({{new_multiset, true}});

                three_multiset.push_back(new_multiset);
            }
        }
    }


    vector<vector<Node>> three_tuples;
    for (vector<Node> ms: three_multiset) {
        three_tuples.push_back({{ms[0], ms[1], ms[2]}});
        three_tuples.push_back({{ms[0], ms[2], ms[1]}});

        three_tuples.push_back({{ms[1], ms[2], ms[0]}});
        three_tuples.push_back({{ms[1], ms[0], ms[2]}});

        three_tuples.push_back({{ms[2], ms[1], ms[0]}});
        three_tuples.push_back({{ms[2], ms[0], ms[1]}});
    }


    size_t num_three_tuples = 0;
    for (vector<Node> tuple: three_tuples) {
        Node i = tuple[0];
        Node j = tuple[1];
        Node k = tuple[2];


        node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
        three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
        num_three_tuples++;

        Label c_i = 1;
        Label c_j = 2;
        Label c_k = 3;


        if (use_labels) {
            c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
            c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);

            c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
        }

        Label a, b, c;
        if (g.has_edge(i, j)) {
            a = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), 1);

            if (use_edge_labels) {
                    a = AuxiliaryMethods::pairing(a, edge_labels.find(make_tuple(i, j))->second);
                }
        } else if (i == j) {
            a = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), 2);
        } else {
            a = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), 4);
        }

        if (g.has_edge(i, k)) {
            b = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_k), 4);

            if (use_edge_labels) {
                b = AuxiliaryMethods::pairing(b, edge_labels.find(make_tuple(i, k))->second);
            }
        } else if (i == k) {
            b = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_k), 5);
        } else {
            b = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_k), 6);
        }

        if (g.has_edge(j, k)) {
            c = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_j, c_k), 7);

            if (use_edge_labels) {
                c = AuxiliaryMethods::pairing(c, edge_labels.find(make_tuple(j, k))->second);
            }
        } else if (j == k) {
            c = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_j, c_k), 8);
        } else {
            c = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_j, c_k), 9);
        }

        Label new_color = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(a, b), c);
        tuple_labels.push_back(new_color);
    }

    return tuple_labels;
}


// Generate node labels for two-tuple graph of graph g.
vector<unsigned long> get_node_labels_3_1(const Graph &g, const bool use_labels, const bool use_edge_labels) {
    size_t num_nodes = g.get_num_nodes();
    // New graph to be generated.

    // Maps node in two set graph to correponding two set.
    unordered_map<Node, ThreeTuple> node_to_three_tuple;
    // Inverse of the above map.
    unordered_map<ThreeTuple, Node> three_tuple_to_node;
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


    // Generate 1-multiset.
    vector<Node> one_multiset;
    for (Node i = 0; i < num_nodes; ++i) {
        one_multiset.push_back(i);
    }

    vector<vector<Node>> two_multiset;

    // Avoid duplicates.
    unordered_map<vector<Node>, bool, VectorHasher> two_multiset_exits;

    for (Node v: one_multiset) {
        for (Node w: g.get_neighbours(v)) {
            vector<Node> new_multiset = {v};
            new_multiset.push_back(w);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = two_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == two_multiset_exits.end()) {
                two_multiset_exits.insert({{new_multiset, true}});

                two_multiset.push_back(new_multiset);
            }
        }

        vector<Node> new_multiset = {v};
        new_multiset.push_back(v);

        std::sort(new_multiset.begin(), new_multiset.end());

        auto t = two_multiset_exits.find(new_multiset);

        // Check if not already exists.
        if (t == two_multiset_exits.end()) {
            two_multiset_exits.insert({{new_multiset, true}});

            two_multiset.push_back(new_multiset);
        }
    }

    vector<vector<Node>> three_multiset;
    unordered_map<vector<Node>, bool, VectorHasher> three_multiset_exits;
    for (vector<Node> ms: two_multiset) {
        for (Node v: ms) {
            for (Node w: g.get_neighbours(v)) {
                vector<Node> new_multiset = {ms[0], ms[1]};
                new_multiset.push_back(w);

                std::sort(new_multiset.begin(), new_multiset.end());

                auto t = three_multiset_exits.find(new_multiset);

                // Check if not already exists.
                if (t == three_multiset_exits.end()) {
                    three_multiset_exits.insert({{new_multiset, true}});

                    three_multiset.push_back(new_multiset);
                }
            }

            vector<Node> new_multiset = {ms[0], ms[1]};
            new_multiset.push_back(v);

            std::sort(new_multiset.begin(), new_multiset.end());

            auto t = three_multiset_exits.find(new_multiset);

            // Check if not already exists.
            if (t == three_multiset_exits.end()) {
                three_multiset_exits.insert({{new_multiset, true}});

                three_multiset.push_back(new_multiset);
            }
        }
    }


    vector<vector<Node>> three_tuples;
    for (vector<Node> ms: three_multiset) {
        three_tuples.push_back({{ms[0], ms[1], ms[2]}});
        three_tuples.push_back({{ms[0], ms[2], ms[1]}});

        three_tuples.push_back({{ms[1], ms[2], ms[0]}});
        three_tuples.push_back({{ms[1], ms[0], ms[2]}});

        three_tuples.push_back({{ms[2], ms[1], ms[0]}});
        three_tuples.push_back({{ms[2], ms[0], ms[1]}});
    }


    size_t num_three_tuples = 0;
    for (vector<Node> tuple: three_tuples) {
        Node i = tuple[0];
        Node j = tuple[1];
        Node k = tuple[2];


        node_to_three_tuple.insert({{num_three_tuples, make_tuple(i, j, k)}});
        three_tuple_to_node.insert({{make_tuple(i, j, k), num_three_tuples}});
        num_three_tuples++;

        Label c_i = 1;
        Label c_j = 2;
        Label c_k = 3;


        if (use_labels) {
            c_i = AuxiliaryMethods::pairing(labels[i] + 1, c_i);
            c_j = AuxiliaryMethods::pairing(labels[j] + 1, c_j);

            c_k = AuxiliaryMethods::pairing(labels[k] + 1, c_k);
        }

        Label a, b, c;
        if (g.has_edge(i, j)) {
            a = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), 1);

            if (use_edge_labels) {
                a = AuxiliaryMethods::pairing(a, edge_labels.find(make_tuple(i, j))->second);
            }
        } else if (i == j) {
            a = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), 2);
        } else {
            a = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_j), 4);
        }

        if (g.has_edge(i, k)) {
            b = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_k), 4);

            if (use_edge_labels) {
                b = AuxiliaryMethods::pairing(b, edge_labels.find(make_tuple(i, k))->second);
            }
        } else if (i == k) {
            b = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_k), 5);
        } else {
            b = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_i, c_k), 6);
        }

        if (g.has_edge(j, k)) {
            c = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_j, c_k), 7);

            if (use_edge_labels) {
                c = AuxiliaryMethods::pairing(c, edge_labels.find(make_tuple(j, k))->second);
            }
        } else if (j == k) {
            c = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_j, c_k), 8);
        } else {
            c = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(c_j, c_k), 9);
        }

        Label new_color = AuxiliaryMethods::pairing(AuxiliaryMethods::pairing(a, b), c);
        tuple_labels.push_back(new_color);
    }

    return tuple_labels;
}


// Generate node labels for two-tuple graph of graph g.
vector<unsigned long> get_node_labels_2_2(const Graph &g, const bool use_labels, const bool use_edge_labels) {
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
        for (Node j = 0; j < num_nodes; ++j) {
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

    return tuple_labels;
}

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
        for (Node j: g.get_neighbours(i)) {
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



// Get all sparse adjacency matrix representations of two-tuple graphs in graph database.
vector<pair<vector<vector<uint>>, vector<vector<uint>>>>
get_all_matrices_2_2(string name, const std::vector<int> &indices) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    GraphDatabase gdb_new;
    for (int i: indices) {
        gdb_new.push_back(gdb[i]);
    }

    vector<pair<vector<vector<uint>>, vector<vector<uint>>>> matrices;
    for (auto &g: gdb_new) {
        matrices.push_back(generate_local_sparse_am_2_2(g));
    }

    return matrices;
}


// Get all sparse adjacency matrix representations of two-tuple graphs in graph database.
vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>>
get_all_matrices_3_2(string name, const std::vector<int> &indices) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    GraphDatabase gdb_new;
    for (int i: indices) {
        gdb_new.push_back(gdb[i]);
    }

    vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>> matrices;
    for (auto &g: gdb_new) {
        matrices.push_back(generate_local_sparse_am_3_2(g));
    }

    return matrices;
}


// Get all sparse adjacency matrix representations of two-tuple graphs in graph database.
vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>>
get_all_matrices_3_1(string name, const std::vector<int> &indices) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);

    GraphDatabase gdb_new;
    for (int i: indices) {
        gdb_new.push_back(gdb[i]);
    }

    vector<tuple<vector<vector<uint>>, vector<vector<uint>>, vector<vector<uint>>>> matrices;
    for (auto &g: gdb_new) {
        matrices.push_back(generate_local_sparse_am_3_1(g));
    }

    return matrices;
}





// Get all node labels of two-tuple graphs in graph database.
vector<vector<unsigned long>> get_all_node_labels_2_2(string name, const bool use_node_labels, const bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);
    vector<vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb) {
        vector<unsigned long> colors = get_node_labels_2_2(g, use_node_labels, use_edge_labels);
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



vector <vector<unsigned long>> get_all_node_labels_allchem_3_2(const bool use_node_labels, const bool use_edge_labels,
                                                           const std::vector<int> &indices_train,
                                                           const std::vector<int> &indices_val,
                                                           const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_val) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_test) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<unsigned long> colors = get_node_labels_3_2(g, use_node_labels, use_edge_labels);
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


vector <vector<unsigned long>> get_all_node_labels_allchem_3_1(const bool use_node_labels, const bool use_edge_labels,
                                                           const std::vector<int> &indices_train,
                                                           const std::vector<int> &indices_val,
                                                           const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_val) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_test) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<unsigned long> colors = get_node_labels_3_1(g, use_node_labels, use_edge_labels);
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


vector <vector<unsigned long>> get_all_node_labels_allchem_2_1(const bool use_node_labels, const bool use_edge_labels,
                                                           const std::vector<int> &indices_train,
                                                           const std::vector<int> &indices_val,
                                                           const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_val) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_test) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
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


vector <vector<unsigned long>> get_all_node_labels_allchem_2_2(const bool use_node_labels, const bool use_edge_labels,
                                                           const std::vector<int> &indices_train,
                                                           const std::vector<int> &indices_val,
                                                           const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("alchemy_full");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_new_1;
    for (auto i : indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_val) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_test) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<unsigned long> colors = get_node_labels_2_2(g, use_node_labels, use_edge_labels);
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


vector <vector<unsigned long>> get_all_node_labels_zinc_3_2(const bool use_node_labels, const bool use_edge_labels,
                                                           const std::vector<int> &indices_train,
                                                           const std::vector<int> &indices_val,
                                                           const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);


    GraphDatabase gdb_new_1;
    for (auto i : indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_val) {
        gdb_new_1.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_test) {
        gdb_new_1.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<unsigned long> colors = get_node_labels_3_2(g, use_node_labels, use_edge_labels);
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


vector <vector<unsigned long>> get_all_node_labels_zinc_3_1(const bool use_node_labels, const bool use_edge_labels,
                                                           const std::vector<int> &indices_train,
                                                           const std::vector<int> &indices_val,
                                                           const std::vector<int> &indices_test) {
    GraphDatabase gdb_1 = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb_1.erase(gdb_1.begin() + 0);

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

    GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);


    GraphDatabase gdb_new_1;
    for (auto i : indices_train) {
        gdb_new_1.push_back(gdb_1[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_val) {
        gdb_new_1.push_back(gdb_2[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    for (auto i : indices_test) {
        gdb_new_1.push_back(gdb_3[int(i)]);
    }
    cout << gdb_new_1.size() << endl;

    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new_1) {
        vector<unsigned long> colors = get_node_labels_3_1(g, use_node_labels, use_edge_labels);
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


vector <vector<unsigned long>> get_all_node_labels_3_2(const string name, const bool use_node_labels, const bool use_edge_labels, const std::vector<int> &indices) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);


    GraphDatabase gdb_new;
    for (auto i : indices) {
        gdb_new.push_back(gdb[int(i)]);
    }
    cout << gdb_new.size() << endl;


    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb_new) {
        vector<unsigned long> colors = get_node_labels_3_2(g, use_node_labels, use_edge_labels);
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


vector <vector<unsigned long>> get_all_node_labels_3_1(const string name, const bool use_node_labels, const bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(name);
    gdb.erase(gdb.begin() + 0);


    vector <vector<unsigned long>> node_labels;

    uint m_num_labels = 0;
    unordered_map<int, int> m_label_to_index;

    for (auto &g: gdb) {
        vector<unsigned long> colors = get_node_labels_3_1(g, use_node_labels, use_edge_labels);
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
    m.def("get_all_matrices_2_2", &get_all_matrices_2_2);
    m.def("get_all_matrices_3_2", &get_all_matrices_3_2);
    m.def("get_all_matrices_3_1", &get_all_matrices_3_1);

    m.def("get_all_node_labels_2_2", &get_all_node_labels_2_2);
    m.def("get_all_node_labels_2_1", &get_all_node_labels_2_1);
    m.def("get_all_node_labels_3_1", &get_all_node_labels_3_1);
    m.def("get_all_node_labels_3_2", &get_all_node_labels_3_2);

    m.def("get_all_attributes_3_2", &get_all_attributes_3_2);
    m.def("get_all_attributes_3_1", &get_all_attributes_3_1);
    m.def("get_all_attributes_2_1", &get_all_attributes_2_1);
    m.def("get_all_attributes_2_2", &get_all_attributes_2_2);

    m.def("get_all_node_labels_allchem_3_2", &get_all_node_labels_allchem_3_2);
    m.def("get_all_node_labels_allchem_3_1", &get_all_node_labels_allchem_3_1);
    m.def("get_all_node_labels_allchem_2_1", &get_all_node_labels_allchem_2_1);
    m.def("get_all_node_labels_allchem_2_2", &get_all_node_labels_allchem_2_2);
    m.def("get_all_node_labels_zinc_3_2", &get_all_node_labels_zinc_3_2);
    m.def("get_all_node_labels_zinc_3_1", &get_all_node_labels_zinc_3_1);

    m.def("read_targets", &read_targets);
}