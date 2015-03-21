import numpy as np
import math
import os


class NN:
    def __init__(self, input_layer, hidden_layer, output_layer, iteration=1000, learning_rate=0.1, momentum=0.9,
                 show_progress_err=False, bold_driver=True, early_stop=2, load_weight=False, save_weight=False):
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

    def train(self, data_set, val_set_percent=25):
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
            if self.early_stop is not False and i > 0:
                k = 10
                gl = (100 * (val_err_err[-1] / np.min(val_err_err) - 1))
                p = 1000 * ((np.sum(train_err_arr[-k:]) / (k * np.min(train_err_arr[-k:]))) - 1)
                pq = gl / p
                if pq > self.early_stop:
                    break
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

    def apply(self, data):
        return self.forward(data)[2][0][0]

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

    def apply(self, data):
        input_matrix = np.append([1, ], data)
        return (input_matrix * self.weight)[0, 0]

    def __str__(self):
        return "Linear Regression"