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
    string ds = "ENZYMES";
    string kernel = "LWL_3_2";
    uint i = 3;
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);

    gdb.erase(gdb.begin() + 0);
    vector<int> classes = AuxiliaryMethods::read_classes(ds);

    GenerateThree::GenerateThree wl(gdb);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    GramMatrix gm = wl.compute_gram_matrix(i, true, "local2", true);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(t2 - t1).count();

    cout << duration << endl;

    AuxiliaryMethods::write_libsvm(gm, classes,
                                   "/Users/chrsmrrs/SeqGN/k_s_wl_cpp/svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                   ".gram");
    cout << "###" << endl;
    return 0;
}
