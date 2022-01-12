import auxiliarymethods.auxiliary_methods as aux
import kernel_baselines as kb
import auxiliarymethods.datasets as dp
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation
import os.path
from os import path as pth
import numpy as np
import time


def main():
    num_reps = 10


    ### Smaller datasets using LIBSVM.
    dataset = [["ENZYMES", True], ["IMDB-BINARY", False], ["NCI1", True],]

    # Number of repetitions of 10-CV.
    num_reps = 10

    results = []
    for dataset, use_labels in dataset:
        classes = dp.get_dataset(dataset)

        # all_matrices = []
        # for i in range(5, 6):
        #     start_time = time.time()
        #     gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
        #     elapsed_time = time.time() - start_time
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "WL_1 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        # results.append(dataset + " " + "WL_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        #
        # all_matrices = []
        # for i in range(5, 6):
        #     start_time = time.time()
        #     gm = kb.compute_lwl_2_dense(dataset, i, use_labels, False)
        #     elapsed_time = time.time() - start_time
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "LWL2 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        # results.append(dataset + " " + "WL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        # all_matrices = []
        # for i in range(5, 6):
        #     start_time = time.time()
        #     gm = kb.compute_lwl_3_dense(dataset, i, use_labels, False)
        #     elapsed_time = time.time() - start_time
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "LWL3 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        # results.append(dataset + " " + "WL3 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        # #
        # #
        # all_matrices = []
        # for i in range(5, 6):
        #     start_time = time.time()
        #     gm = kb.compute_lwlp_2_dense(dataset, i, use_labels, False)
        #     elapsed_time = time.time() - start_time
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "LWLP2 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        # results.append(dataset + " " + "WL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        # all_matrices = []
        # for i in range(5, 6):
        #     start_time = time.time()
        #     gm = kb.compute_lwlp_3_dense(dataset, i, use_labels, False)
        #     elapsed_time = time.time() - start_time
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "LWLP3 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        # results.append(dataset + " " + "WL3 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        #
        # all_matrices = []
        # for i in range(5, 6):
        #     start_time = time.time()
        #     gm = kb.compute_wl_2_dense(dataset, i, use_labels, False)
        #     elapsed_time = time.time() - start_time
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "WL2 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        # results.append(dataset + " " + "WL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        #
        # all_matrices = []
        # for i in range(5, 6):
        #     start_time = time.time()
        #     gm = kb.compute_wl_3_dense(dataset, i, use_labels, False)
        #     elapsed_time = time.time() - start_time
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "WL3 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        # results.append(dataset + " " + "WL3 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(5, 6):
            start_time = time.time()
            gm = kb.compute_wlp_2_1_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WLP2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WLP2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        all_matrices = []
        for i in range(5, 6):
            start_time = time.time()
            gm = kb.compute_wl_3_1_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WL3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(5, 6):
            start_time = time.time()
            gm = kb.compute_wlp_3_1_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WLP3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WLP3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # all_matrices = []
        # for i in range(1, 6):
        #     gm = kb.compute_wl_3_2_dense(dataset, i, use_labels, False)
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "WL3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        # results.append(dataset + " " + "WL3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        # all_matrices = []
        # for i in range(1, 6):
        #     gm = kb.compute_wlp_3_2_dense(dataset, i, use_labels, False)
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "WLP3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        # results.append(dataset + " " + "WLP3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # # WLOA kernel, number of iterations in [1:6].
        # all_matrices = []
        # for i in range(1, 6):
        #     gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
        #     gm_n = aux.normalize_gram_matrix(gm)
        #     all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "WLOA " + str(acc) + " " + str(s_1) + " " + str(s_2))
        # results.append(dataset + " " + "WLOA " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        # # Graphlet kernel.
        # all_matrices = []
        # gm = kb.compute_graphlet_dense(dataset, use_labels, False)
        # gm_n = aux.normalize_gram_matrix(gm)
        # all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "GR " + str(acc) + " " + str(s_1) + " " + str(s_2))
        # results.append(dataset + " " + "GR " + str(acc) + " " + str(s_1) + " " + str(s_2))
        #
        # # Shortest-path kernel.
        # all_matrices = []
        # gm = kb.compute_shortestpath_dense(dataset, use_labels)
        # gm_n = aux.normalize_gram_matrix(gm)
        # all_matrices.append(gm_n)
        # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        # print(dataset + " " + "SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        # results.append(dataset + " " + "SP " + str(acc) + " " + str(s_1) + " " + str(s_2))


    return 0
    # Number of repetitions of 10-CV.
    num_reps = 3

    ### Larger datasets using LIBLINEAR with edge labels.
    dataset = [["MOLT-4", True, True], ["Yeast", True, True], ["MCF-7", True, True],
               ["github_stargazers", False, False],
               ["reddit_threads", False, False]]

    for d, use_labels, use_edge_labels in dataset:
        dataset = d
        classes = dp.get_dataset(dataset)

        # 1-WL kernel, number of iterations in [1:6].
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_sparse(dataset, i, use_labels, use_edge_labels)
            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)

        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(d + " " + "WL1SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "WL1SP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # 1-WL kernel, number of iterations in [1:6].
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_2_1_sparse(dataset, i, use_labels, use_edge_labels)
            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)

        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(d + " " + "WL21SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "WL21SP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # Graphlet kernel, number of iterations in [1:6].
        all_matrices = []
        gm = kb.compute_graphlet_sparse(dataset, use_labels, use_edge_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)

        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GRSP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GRSP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # Shortest-path kernel.
        all_matrices = []
        gm = kb.compute_shortestpath_sparse(dataset, use_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)

        acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(d + " " + "SPSP " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "SPSP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        for r in results:
            print(r)



if __name__ == "__main__":
    main()
