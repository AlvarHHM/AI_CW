from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import csv
import multiprocessing
import random


def main():
    np.seterr(over='raise')
    data = read_data_set('pre_processed_data.csv')
    # nn_learn_provider = lambda: NN(8, 6, 1, iteration=1000)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data)).start()
    nn_learn_provider = lambda: NN(8, 6, 1, iteration=10000)
    split_set_val(nn_learn_provider, data, write_result=True)


def plot_result_graph(predicted_set, target_set):
    predicted_set = [x for x, y in sorted(zip(predicted_set, target_set), key=lambda pair:pair[1])]
    target_set = [y for x, y in sorted(zip(predicted_set, target_set), key=lambda pair:pair[1])]
    ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
    ax.set_ylabel("Value")
    ax.set_xlabel("Data point")
    target_line, = ax.plot(target_set, label="Target")
    predicted_line, = ax.plot(predicted_set, label="Predicted")
    ax.legend(handler_map={target_line: HandlerLine2D(numpoints=4)})
    ax.legend(handler_map={predicted_line: HandlerLine2D(numpoints=4)})

    ax = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
    ax.set_ylabel("Observed data")
    ax.set_xlabel("Modelled data")
    ax.plot(predicted_set, target_set, 'ro')
    ax.plot(ax.get_xlim(), ax.get_ylim(),)
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def write_csv_result(learner, data, name="result.csv"):
    with open(name, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow([row[1], learner.apply(row[0])])


def split_set_val(learn_provider, data_set, split_percent=20,
                  show_process_graph=False, show_result_graph=False, write_result=False):
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
    if show_process_graph:
        if len(progress_err) != 0:
            job_learning_rate = multiprocessing.Process(target=plot_learning_rate,
                                                        args=(progress_err, val_err))
            job_learning_rate.start()
    if show_result_graph:
        predicted = [network.apply(x[0]) for x in test_set]
        target = [x[1] for x in test_set]
        plot_result_graph(predicted, target)
        job_result_graph = multiprocessing.Process(target=plot_result_graph,
                                                   args=(predicted, target))
        job_result_graph.start()
    if write_result:
        write_csv_result(network, test_set, name="result-"+str(test_err[0])+".csv")
    return test_err


def k_fold_val(learn_provider, data_set, fold=10):
    error = 0
    np.random.shuffle(data_set)
    for i in range(0, fold):
        network = learn_provider()
        train_set = [data_set[index] for index in range(len(data_set)) if ((index - i) % fold) != 0]
        test_set = data_set[i::fold]
        network.train(train_set)
        error += network.test(test_set)[0]
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