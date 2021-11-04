#include <cstdio>
#include "src/AuxiliaryMethods.h"
#include "src/ColorRefinementKernel.h"
#include "src/GenerateTwo.h"
#include "src/GenerateThree.h"
#include "src/Graph.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace std;
using namespace GraphLibrary;
using namespace std;




int
main() {
// k = 2.
    {
        vector<tuple<string, bool, bool>> datasets = {make_tuple("Yeast", true, true)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);
                bool use_edge_labels = std::get<2>(d);

                string kernel = "LWL2_1";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 2; i <= 2; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "local1", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "local1", true);
                    }

                    AuxiliaryMethods::write_sparse_gram_matrix(gm,"/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXPSPARSE/" + ds + "__" + kernel + "_" + to_string(i));
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);
                bool use_edge_labels = std::get<2>(d);

                string kernel = "LWLP2_1";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "local1p", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, use_edge_labels, "local1p", true);
                    }

                    AuxiliaryMethods::write_sparse_gram_matrix(gm,"/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXPSPARSE/" + ds + "__" + kernel + "_" + to_string(i));
                }
            }
        }
    }


    return 0;
}
