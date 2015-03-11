import numpy as np
import math
import matplotlib.pyplot as plt
import csv


class NN:
    def __init__(self, input_layer, hidden_layer, output_layer=1):
        self.weight = [[], []]
        self.weight[0] = np.random.uniform(-1 * 2.0 / input_layer, 2.0 / input_layer,
                                           (hidden_layer, input_layer + 1))
        self.weight[1] = np.random.uniform(-1 * 2.0 / hidden_layer, 2.0 / hidden_layer,
                                           (output_layer, hidden_layer + 1))
        self.ihwc = np.zeros((input_layer, hidden_layer + 1))
        self.howc = np.zeros((hidden_layer, output_layer + 1))
        self.learning_rate = 0.1
        self.momentum = 0.0

    def forward(self, train_data):
        input_neuron_output = np.matrix(train_data).transpose().getA()
        input_neuron_output = np.append([[1]], input_neuron_output, axis=0)

        z = np.dot(self.weight[0], input_neuron_output)
        hidden_neuron_output = self.sigmoid(z)
        hidden_neuron_output = np.append([[1]], hidden_neuron_output, axis=0)

        z = np.dot(self.weight[1], hidden_neuron_output)
        output_neuron_output = self.sigmoid(z)
        return input_neuron_output, hidden_neuron_output, output_neuron_output

    def backward(self, train_output, target):
        input_neuron_output, hidden_neuron_output, output_neuron_output = train_output
        output_delta = (target - output_neuron_output)
        hidden_delta = self.weight[1].transpose() .dot(output_delta) .dot(self.dsigmoid(hidden_neuron_output))
        self.weight[1] += self.learning_rate * output_delta * hidden_neuron_output
        self.weight[0] += self.learning_rate * hidden_delta * input_neuron_output
        1==1
        # output_delta = (target - train_output[2]) * (self.dsigmoid(train_output[2]))
        # hidden_delta = np.zeros(self.hl)
        # for j in range(self.hl):
        #     hidden_delta[j] = self.how[j] * output_delta * self.dsigmoid(train_output[1][j])
        #
        # # update weight from hidden to output
        # # update bias weight
        # self.how[-1] += self.learning_rate * output_delta
        # self.howc[-1] = self.learning_rate * output_delta
        # for j in range(self.hl):
        #     self.how[j] += self.learning_rate * output_delta * train_output[1][j] \
        #                    + self.momentum * self.howc[j]
        #     self.howc[j] = self.learning_rate * output_delta * train_output[1][j]
        #
        # # update weight from input to hidden
        # for j in range(self.hl):
        #     # update bias weight
        #     self.ihw[-1, j] += self.learning_rate * hidden_delta[j]
        #     self.ihwc[-1, j] = self.learning_rate * hidden_delta[j]
        #     for i in range(self.il):
        #         self.ihw[i, j] += self.learning_rate * hidden_delta[j] * train_output[0][i] \
        #                           + self.momentum * self.ihwc[i, j]
        #         self.ihwc[i, j] = self.learning_rate * hidden_delta[j] * train_output[0][i]
        #
        # # return calculate error
        # return (target - train_output[2]) ** 2

    def train(self, data, iteration=10000):
        err_arr = []
        for i in range(iteration):
            error = 0.0
            for datum in data:
                output = self.forward(datum[0])
                error += self.backward(output, datum[1])
            if i % 100 == 0:
                err_arr.append(error)
                print error
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')
        ax.plot(err_arr)
        plt.show()

        np.save("ihw", self.ihw)
        np.save("how", self.how)

    def test(self, test_set):
        err_arr = []
        for datum in test_set:
            error = (datum[1] - self.forward(datum[0])) ** 2
            err_arr.append(error)
        return math.sqrt(reduce(lambda x, y: x + y, err_arr) / len(err_arr))


    @staticmethod
    def matrix_multi(x, y):
        return (x[:, :, None] * y).sum(axis=1)

    @staticmethod
    def dsigmoid(x):
        # return 1.0 - x ** 2
        ans = x * (1 - x)
        return ans

    @staticmethod
    def sigmoid(x):
        # return math.tanh(x)
        # return 1 / (1 + math.exp(-x))
        # ans = 1/2*(1+math.tanh(x/2))
        return 0.5 * (1. + np.tanh(x / 2.))
        # try:
        # return 1 / (1 + math.exp(-x))
        # except OverflowError:
        #     if x > 0:
        #         return 1
        #     else:
        #         return 0


def demo():
    pat = [
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0)
    ]
    n = NN(2, 5)
    # train it with some patterns
    n.train(pat)
    # test it
    # n.test(pat)


def readDataFromFile(file):
    data = []
    with open(file, 'rbU') as csvfile:
        spamreader = csv.reader(csvfile, dialect=csv.excel_tab)
        for row in spamreader:
            row_data = [float(x) for x in row[0].split(',')]
            data.append((row_data[:-1], row_data[-1]))
    return data


def main():
    np.seterr(over='raise')
    data = readDataFromFile('CWDatav6.csv')
    n = NN(8, 3)
    # print data
    n.train(data)


if __name__ == '__main__':
    demo()
