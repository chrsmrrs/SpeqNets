import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation


def main():
    ### Smaller datasets using LIBSVM.
    dataset = [["IMDB-BINARY", False]]

    # Number of repetitions of 10-CV.
    num_reps = 10

    results = []
    for dataset, use_labels in dataset:
        classes = dp.get_dataset(dataset)

        # 1-WL kernel, number of iterations in [1:6].
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_2_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
        print(dataset + " " + "WL1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(dataset + " " + "WL1 " + str(acc) + " " + str(s_1) + " " + str(s_2))



if __name__ == "__main__":
    main()
