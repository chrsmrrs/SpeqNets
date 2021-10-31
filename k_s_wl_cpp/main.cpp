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
        vector<pair<string, bool>> datasets = {make_pair("ENZYMES", true), make_pair("IMDB-BINARY", false),
                                               make_pair("IMDB-MULTI", false), make_pair("NCI1", true),
                                               make_pair("NCI109", true), make_pair("PTC_FM", true),
                                               make_pair("PROTEINS", true), make_pair("REDDIT-BINARY", false)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWL2_1";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local1", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local1", true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

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
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local1p", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local1p", true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }


    // k = 3.
    {
        vector<pair<string, bool>> datasets = {make_pair("ENZYMES", true), make_pair("IMDB-BINARY", false),
                                               make_pair("IMDB-MULTI", false), make_pair("NCI1", true),
                                               make_pair("NCI109", true), make_pair("PTC_FM", true),
                                               make_pair("PROTEINS", true), make_pair("REDDIT-BINARY", false)};

        for (auto &d: datasets) {
            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWL3_1";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "local1", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "local1", true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWL3_2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "local2", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "local2", true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWLP3_1";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "local1p", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "local1p", true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string ds = std::get<0>(d);
                bool use_labels = std::get<1>(d);

                string kernel = "LWLP3_2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "local2p", true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "local2p", true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    return 0;
}
