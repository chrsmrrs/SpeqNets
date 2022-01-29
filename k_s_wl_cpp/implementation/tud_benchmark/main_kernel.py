import time

import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation


def main():
    num_reps = 10

    dataset = [["ENZYMES", True], ["IMDB-BINARY", False], ["NCI1", True], ["PROTEINS", True]]

    # Number of repetitions of 10-CV.
    num_reps = 10

    results = []
    for dataset, use_labels in dataset:
        classes = dp.get_dataset(dataset)

        all_matrices = []
        for i in range(1, 6):
            start_time = time.time()
            gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL_1 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WL_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))


        all_matrices = []
        for i in range(1, 6):
            start_time = time.time()
            gm = kb.compute_wl_2_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL2 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))


        all_matrices = []
        for i in range(1, 6):
            start_time = time.time()
            gm = kb.compute_lwl_2_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "LWL2 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "LWL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))


        all_matrices = []
        for i in range(1, 6):
            start_time = time.time()
            gm = kb.compute_lwlp_2_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "LWLP2 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WL2 " + str(acc) + " " + str(s_1) + " " + str(s_2))


        all_matrices = []
        for i in range(1, 6):
            start_time = time.time()
            gm = kb.compute_wl_2_1_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WL2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            start_time = time.time()
            gm = kb.compute_wlp_2_1_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WLP2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WLP2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # WLOA kernel, number of iterations in [1:6].
        all_matrices = []
        for i in range(1, 6):
            start_time = time.time()
            gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
            elapsed_time = time.time() - start_time
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WLOA " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "WLOA " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # Graphlet kernel.
        all_matrices = []
        start_time = time.time()
        gm = kb.compute_graphlet_dense(dataset, use_labels, False)
        elapsed_time = time.time() - start_time
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "GR " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "GR " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # Shortest-path kernel.
        all_matrices = []
        start_time = time.time()
        gm = kb.compute_shortestpath_dense(dataset, use_labels)
        elapsed_time = time.time() - start_time
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "SP " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(elapsed_time))
        results.append(dataset + " " + "SP " + str(acc) + " " + str(s_1) + " " + str(s_2))

        for r in results:
            print(r)


if __name__ == "__main__":
    main()
