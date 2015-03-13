import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import os
import multiprocessing


class NN:
    def __init__(self, input_layer, hidden_layer, output_layer, iteration=1000, learning_rate=0.1, momentum=0.9,
                 show_progress_err=False, load_weight=False):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.show_progress_err = show_progress_err
        self.weight = [[], []]
        self.weight_change = [[], []]
        if load_weight & os.path.exists("weight.npy"):
            self.weight = np.load("weight.npy")
        else:
            self.weight[0] = np.random.uniform(-1 * 2.0 / input_layer, 2.0 / input_layer,
                                               (hidden_layer, input_layer + 1))
            self.weight[1] = np.random.uniform(-1 * 2.0 / hidden_layer, 2.0 / hidden_layer,
                                               (output_layer, hidden_layer + 1))
        self.weight_change[0] = np.zeros((hidden_layer, input_layer + 1))
        self.weight_change[1] = np.zeros((output_layer, hidden_layer + 1))

    def forward(self, train_data):
        input_neuron_output = np.matrix(train_data).transpose().getA()
        input_neuron_output = np.append([[1]], input_neuron_output, axis=0)

        z = np.dot(self.weight[0], input_neuron_output)
        hidden_neuron_output = self.sigmoid(z)
        hidden_neuron_output = np.append([[1]], hidden_neuron_output, axis=0)

        z = np.dot(self.weight[1], hidden_neuron_output)
        output_neuron_output = self.sigmoid(z)
        return input_neuron_output, hidden_neuron_output, output_neuron_output

    def backward(self, train_output, target, learning_rate=0.1, momentum=0.9):
        input_neuron_output, hidden_neuron_output, output_neuron_output = train_output
        output_delta = (np.matrix(target).transpose().getA() - output_neuron_output)
        hidden_delta = self.weight[1].transpose().dot(output_delta) * (self.dsigmoid(hidden_neuron_output))

        change = output_delta * np.matrix(hidden_neuron_output).transpose()
        self.weight[1] += learning_rate * change + momentum * self.weight_change[1]
        self.weight_change[1] = change

        change = hidden_delta[1:] * np.matrix(input_neuron_output).transpose()
        self.weight[0] += learning_rate * change + momentum * self.weight_change[0]
        self.weight_change[0] = change
        return output_delta

    def train(self, data):
        err_arr = []
        for i in range(self.iteration):
            error = 0.0
            for datum in data:
                output = self.forward(datum[0])
                error += self.backward(output, datum[1], self.learning_rate, self.momentum)
            error = np.matrix(error).getA1() ** 2
            err_arr.append(error[0])
            if self.show_progress_err:
                print "error at " + str(i) + " iteration: " + str(error)
        np.save("weight", self.weight)
        return err_arr

    def test(self, test_set):
        err_arr = []
        for datum in test_set:
            error = (datum[1] - self.forward(datum[0])[2][0]) ** 2
            err_arr.append(error)
        return math.sqrt(reduce(lambda x, y: x + y, err_arr) / len(err_arr))

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
        return (self.target_matrix - (self.input_matrix * self.weight)).mean()


def main():
    np.seterr(over='raise')
    data = read_data_set('CWDatav6.csv')
    nn_learn_provider = lambda: NN(8, 5, 1, iteration=100, learning_rate=0.1, momentum=0.9, load_weight=False)
    lr_learn_provider = lambda: LR()
    # multiprocessing.Process(target=split_set_val, args=(nn_learn_provider, data, 25)).start()
    multiprocessing.Process(target=k_fold_val, args=(nn_learn_provider, data, 10)).start()
    # multiprocessing.Process(target=split_set_val, args=(lr_learn_provider, data, 25)).start()
    # multiprocessing.Process(target=k_fold_val, args=(lr_learn_provider, data, 10)).start()

    # split_set_val(learn_provider, data, split_percent=25)
    # k_fold_val(learn_provider, data, fold=10)


def split_set_val(learn_provider, data_set, split_percent=25):
    network = learn_provider()
    np.random.shuffle(data_set)
    train_set = data_set[int(len(data_set) * (split_percent / 100.0)):]
    test_set = data_set[:int(len(data_set) * (split_percent / 100.0))]
    progress_err = network.train(train_set)
    print "Error on test set:  " + str(network.test(test_set))
    if len(progress_err) != 0:
        job_for_another_core = multiprocessing.Process(target=plot_learning_rate, args=(progress_err,))
        job_for_another_core.start()


def k_fold_val(learn_provider, data_set, fold=10):
    error = 0
    for i in range(0, fold):
        network = learn_provider()
        train_set = [data_set[index] for index in range(len(data_set)) if ((index - i) % fold) != 0]
        test_set = data_set[i::fold]
        network.train(train_set)
        error += network.test(test_set)
    print "Average error on K-fold: " + str(error / fold)


def plot_learning_rate(err_arr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    plt.title("Error against iteration")
    plt.ylabel("Square error")
    plt.xlabel("Iteration")
    ax.plot(err_arr)
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
    demo()
