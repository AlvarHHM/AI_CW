import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import csv
import os
import multiprocessing
import random


class NN:
    def __init__(self, input_layer, hidden_layer, output_layer, iteration=1000, learning_rate=0.1, momentum=0.9,
                 show_progress_err=False, bold_driver=True, early_stop=True, load_weight=False, save_weight=False):
        self.input_layer, self.hidden_layer, self.output_layer = input_layer, hidden_layer, output_layer
        self.iteration, self.learning_rate, self.momentum = iteration, learning_rate, momentum
        self.bold_driver, self.early_stop = bold_driver, early_stop
        self.save_weight = save_weight
        self.show_progress_err = show_progress_err
        self.weight = [[], []]
        self.weight_change = [[], []]
        if load_weight & os.path.exists("weight.npy"):
            self.weight = np.load("weight.npy")
        else:
            # Weight matrix from layer 1(input) to layer 2(hidden)
            self.weight[0] = np.random.uniform(-1 * 2.0 / input_layer, 2.0 / input_layer,
                                               (hidden_layer, input_layer + 1))
            # Weight matrix from layer 2(hidden) to layer 3(output)
            self.weight[1] = np.random.uniform(-1 * 2.0 / hidden_layer, 2.0 / hidden_layer,
                                               (output_layer, hidden_layer + 1))
        self.weight_change[0] = np.zeros((hidden_layer, input_layer + 1))
        self.weight_change[1] = np.zeros((output_layer, hidden_layer + 1))
        self.backup_weight = self.weight
        self.backup_weight_change = self.weight_change

    def forward(self, train_data):
        # output vector of layer 1 is input attribute
        input_neuron_output = np.matrix(train_data).transpose().getA()
        input_neuron_output = np.append([[1]], input_neuron_output, axis=0)  # append 1 as bias

        z = np.dot(self.weight[0], input_neuron_output)  # multiply weight and output vector of layer 1
        hidden_neuron_output = self.sigmoid(z)  # apply sigmoid function to the output vector
        hidden_neuron_output = np.append([[1]], hidden_neuron_output, axis=0)  # append 1 as bias

        z = np.dot(self.weight[1], hidden_neuron_output)  # multiply weight by output vector of layer 2
        output_neuron_output = self.sigmoid(z)  # apply sigmoid function to the output vector
        return input_neuron_output, hidden_neuron_output, output_neuron_output

    def backward(self, train_output, target, learning_rate=0.1, momentum=0.9):
        input_neuron_output, hidden_neuron_output, output_neuron_output = train_output
        output_delta = (np.matrix(target).transpose().getA() - output_neuron_output)
        hidden_delta = np.dot(self.weight[1].transpose(), output_delta) * (self.dsigmoid(hidden_neuron_output))

        change = output_delta * np.matrix(hidden_neuron_output).transpose()
        self.weight[1] += learning_rate * change + momentum * self.weight_change[1]
        self.weight_change[1] = change

        change = hidden_delta[1:] * np.matrix(input_neuron_output).transpose()
        self.weight[0] += learning_rate * change + momentum * self.weight_change[0]
        self.weight_change[0] = change
        return output_delta

    def train(self, data_set, val_set_percent=20):
        train_set = data_set[:int(val_set_percent / 100.0 * len(data_set))]
        val_set = data_set[int(val_set_percent / 100.0 * len(data_set)):]
        train_err_arr, val_err_err = [], []
        for i in range(self.iteration):
            error = 0.0
            for datum in train_set:
                output = self.forward(datum[0])
                error += self.backward(output, datum[1], self.learning_rate, self.momentum) ** 2
            error = np.sqrt(np.matrix(error).getA1() / len(train_set))
            if self.bold_driver:
                if i > 0 and error > train_err_arr[-1]:
                    self.learning_rate *= 0.5
                    self.restore_snapshot()
                    continue
                else:
                    self.learning_rate *= 1.1
                    self.snapshot()
            train_err_arr.append(error[0])
            val_err_err.append(self.test(val_set))
            if self.early_stop and i > 0:
                k = 10
                pq = (100 * (val_err_err[-1] / np.min(val_err_err) - 1)) \
                    / ((np.sum(train_err_arr[-k:]) / (k * np.min(train_err_arr[-k:]))) - 1)
                if pq > 0:
                    pass
                    # break
            if self.show_progress_err:
                print "error at " + str(i) + " iteration: " + str(error)
        if self.save_weight:
            np.save("weight", self.weight)
        return train_err_arr, val_err_err

    def test(self, test_set):
        err_arr = []
        for datum in test_set:
            error = (datum[1] - self.forward(datum[0])[2][0]) ** 2
            err_arr.append(error)
        return math.sqrt(reduce(lambda x, y: x + y, err_arr) / len(err_arr))

    def snapshot(self):
        self.backup_weight = self.weight
        self.backup_weight_change = self.weight_change

    def restore_snapshot(self):
        self.weight = self.backup_weight
        self.weight_change = self.backup_weight_change

    def __str__(self):
        return "NN with %d hidden, %d iteration, %0.3f rate, %0.3f momentum, %r bold_driver, %r early stop" \
               % (self.hidden_layer, self.iteration, self.learning_rate, self.momentum,
                  self.bold_driver, self.early_stop)

    @staticmethod
    def dsigmoid(x):
        # return 1.0 - x ** 2
        ans = x * (1 - x)
        return ans

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class LR:
    def __init__(self):
        self.input_matrix = []
        self.target_matrix = []
        self.weight = []
        return

    def train(self, data_set):
        self.input_matrix = np.matrix([np.append([1, ], x[0]) for x in data_set])
        self.target_matrix = np.matrix([x[1] for x in data_set]).transpose()
        self.weight = (self.input_matrix.transpose() * self.input_matrix).getI() \
                      * self.input_matrix.transpose() * self.target_matrix
        return []

    def test(self, data_set):
        self.input_matrix = np.matrix([np.append([1, ], x[0]) for x in data_set])
        self.target_matrix = np.matrix([x[1] for x in data_set]).transpose()
        return math.sqrt((np.array(self.target_matrix - (self.input_matrix * self.weight)) ** 2).mean())

    def __str__(self):
        return "Linear Regression"


def main():
    np.seterr(over='raise')
    data = read_data_set('CWDatav6.csv')

    # test_bold_driver(data)

    # nn_learn_provider = lambda: NN(8, 4, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 20), kwargs={"show_graph": True}).start()
    # nn_learn_provider = lambda: NN(8, 4, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25), kwargs={"show_graph": True}).start()
    # nn_learn_provider = lambda: NN(8, 6, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25)).start()
    # nn_learn_provider = lambda: NN(8, 8, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25)).start()
    # nn_learn_provider = lambda: NN(8, 10, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25)).start()
    # nn_learn_provider = lambda: NN(8, 12, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25)).start()
    # nn_learn_provider = lambda: NN(8, 14, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25)).start()
    # nn_learn_provider = lambda: NN(8, 16, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25)).start()

    # nn_learn_provider = lambda: NN(8, 2, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 4, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 6, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 8, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 10, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 12, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 14, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 16, 1, iteration=1000, bold_driver=True, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()

    nn_learn_provider = lambda: NN(8, 5, 1, iteration=1000, bold_driver=True, early_stop=False)
    multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    nn_learn_provider = lambda: NN(8, 5, 1, iteration=1000, bold_driver=False, early_stop=False)
    multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 5, 1, iteration=1000, bold_driver=False, early_stop=True)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()
    # nn_learn_provider = lambda: NN(8, 5, 1, iteration=1000, bold_driver=False, early_stop=False)
    # multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 3)).start()


    lr_learn_provider = lambda: LR()
    k_fold_val(lr_learn_provider,data,fold=10)
    # multiprocessing.Process(target=split_set_val, args=(lr_learn_provider, data, 25)).start()
    # for i in range(10):
    #     nn_learn_provider = lambda: NN(8, 5, 1, iteration=1000, bold_driver=False, early_stop=True)
    #     split_set_val(nn_learn_provider, data, split_percent=25)
    # k_fold_val(learn_provider, data, fold=10)


def test_bold_driver(data):
    jobs = []
    manager = multiprocessing.Manager()
    no_bold_list = manager.list()
    yes_bold_list = manager.list()
    for i in range(100):
        nn_learn_provider = lambda: NN(8, 5, 1, iteration=1000, bold_driver=False, early_stop=True)
        j1 = multiprocessing.Process(target=test_bold_driver_worker, args=(no_bold_list, nn_learn_provider, data, 25))
        j1.start()
        nn_learn_provider = lambda: NN(8, 5, 1, iteration=1000, bold_driver=False, early_stop=False)
        j2 = multiprocessing.Process(target=test_bold_driver_worker, args=(yes_bold_list, nn_learn_provider, data, 25))
        j2.start()
        jobs.append(j1)
        jobs.append(j2)
    for j in jobs:
        j.join()
    print "without bold driver " + str(np.array(no_bold_list).mean())
    print "with bold driver " + str(np.array(yes_bold_list).mean())


def test_bold_driver_worker(err_list, learn_provider, data_set, split_percent=25):
    err_list.append(split_set_val(learn_provider, data_set, split_percent))


def split_set_val(learn_provider, data_set, split_percent=25, show_graph=False):
    random.seed()
    network = learn_provider()
    np.random.shuffle(data_set)
    train_set = data_set[int(len(data_set) * (split_percent / 100.0)):]
    test_set = data_set[:int(len(data_set) * (split_percent / 100.0))]
    err = network.train(train_set)
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


def demo():
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]]

    n = NN(2, 3, 1, iteration=1000)
    # train it with some patterns
    n.train(pat)
    # test it
    print n.test(pat)


if __name__ == '__main__':
    main()
