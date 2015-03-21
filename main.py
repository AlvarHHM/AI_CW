from VectorNN import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import csv
import multiprocessing
import random


def main():
    np.seterr(over='raise')
    data = read_data_set('CWDatav6.csv')

    # lr_learn_provider = lambda: LR()
    # nn_learn_provider = lambda: NN(8, 6, 1, iteration=10000, load_weight=True)
    # write_result(lr_learn_provider, data)

    nn_learn_provider = lambda: NN(8, 6, 1, iteration=10000, learning_rate=0.01)
    multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data)).start()
    nn_learn_provider = lambda: NN(8, 6, 1, iteration=10000, learning_rate=0.1)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data), kwargs={"show_graph": False}).start()
    multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data)).start()
    #
    # lr_learn_provider = lambda: LR()
    # k_fold_val(lr_learn_provider, data, fold=10)
    # split_set_val(lr_learn_provider, data)


def write_result(learn_provider, data):
    learner = learn_provider()
    # learner.train(data)
    with open('result.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow([row[1], learner.apply(row[0])])


def split_set_val(learn_provider, data_set, split_percent=20, show_graph=False):
    random.seed()
    network = learn_provider()
    np.random.shuffle(data_set)
    train_set = data_set[int(len(data_set) * (split_percent / 100.0)):]
    test_set = data_set[:int(len(data_set) * (split_percent / 100.0))]
    err = network.train(train_set)
    if len(err) == 2:
        progress_err = err[0]
        val_err = err[1]
    test_err = network.test(test_set)
    print str(network)
    print "Error on test set:  " + str(test_err)
    if show_graph:
        if len(progress_err) != 0:
            job_for_another_core = multiprocessing.Process(target=plot_learning_rate,
                                                           args=(progress_err, val_err))
            job_for_another_core.start()
    return test_err


def k_fold_val(learn_provider, data_set, fold=10):
    error = 0
    np.random.shuffle(data_set)
    for i in range(0, fold):
        network = learn_provider()
        train_set = [data_set[index] for index in range(len(data_set)) if ((index - i) % fold) != 0]
        test_set = data_set[i::fold]
        network.train(train_set)
        error += network.test(test_set)
    print str(network)
    print "Average error on K-fold: " + str(error / fold)


def plot_learning_rate(train_set_err, val_set_err, title="error against iteration"):
    plt.title(title)
    plt.ylabel("Square error")
    plt.xlabel("Iteration")
    train_line, = plt.plot(train_set_err, label="Training set error")
    val_line, = plt.plot(val_set_err, label="Validation set error")
    plt.legend(handler_map={train_line: HandlerLine2D(numpoints=4)})
    plt.legend(handler_map={val_line: HandlerLine2D(numpoints=4)})
    plt.show()


def read_data_set(input_file):
    data = []
    with open(input_file, 'rbU') as csv_file:
        reader = csv.reader(csv_file, dialect=csv.excel_tab)
        for row in reader:
            row_data = [float(x) for x in row[0].split(',')]
            data.append((row_data[:-1], row_data[-1]))
    return data


if __name__ == '__main__':
    main()