import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation


def main():
    ### Smaller datasets using LIBSVM.
    dataset = [["IMDB-BINARY", False], ["ENZYMES", True], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
               ["REDDIT-BINARY", False], ["PTC_MR", True],["MUTAG", True]]

    # Number of repetitions of 10-CV.
    num_reps = 10

    results = []
    for dataset, use_labels in dataset:
        classes = dp.get_dataset(dataset)

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(dataset + " " + "WL_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))


        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_2_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(dataset + " " + "WL2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wlp_2_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WLP2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(dataset + " " + "WLP2_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_3_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(dataset + " " + "WL3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

    all_matrices = []
    for i in range(1, 6):
        gm = kb.compute_wlp_3_1_dense(dataset, i, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
    acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print(dataset + " " + "WLP3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
    results.append(dataset + " " + "WLP3_1 " + str(acc) + " " + str(s_1) + " " + str(s_2))

    all_matrices = []
    for i in range(1, 6):
        gm = kb.compute_wl_3_2_dense(dataset, i, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
    acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print(dataset + " " + "WL3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
    results.append(dataset + " " + "WL3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))

    all_matrices = []
    for i in range(1, 6):
        gm = kb.compute_wlp_3_2_dense(dataset, i, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
    acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
    print(dataset + " " + "WLP3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))
    results.append(dataset + " " + "WLP3_2 " + str(acc) + " " + str(s_1) + " " + str(s_2))






if __name__ == "__main__":
    main()
