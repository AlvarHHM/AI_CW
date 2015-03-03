import numpy as np
import math


class NN:
    def __init__(self, input_layer, hidden_layer, no_example):
        self.il = input_layer
        self.hl = hidden_layer
        #self.ihw = np.random.uniform(-1 * 2 / no_example, 2 / no_example, [self.il + 1, self.hl])
        #self.how = np.random.uniform(-1 * 2 / no_example, 2 / no_example, [self.hl + 1])
        self.ihw = np.random.uniform(-0.2, 0.2, [self.il + 1, self.hl])
        self.how = np.random.uniform(-0.2, 0.2, [self.hl + 1])
        self.alpha = 0.1

    def forward(self, train_data):
        hidden_neuron_output = np.zeros(self.hl)
        for hidden_neuron in range(self.hl):
            output_sum = 0
            output_sum += self.ihw[-1, hidden_neuron]
            for input_neuron in range(self.il):
                output_sum += train_data[input_neuron] * self.ihw[input_neuron, hidden_neuron]
            hidden_neuron_output[hidden_neuron] = self.sigmoid(output_sum)

        output_sum = 0
        output_sum += self.how[-1]
        for hidden_neuron in range(self.hl):
            output_sum += hidden_neuron_output[hidden_neuron] * self.how[hidden_neuron]
        output_neuron_output = self.sigmoid(output_sum)

        return hidden_neuron_output, output_neuron_output

    def backward(self, train_output, target):
        output_delta = (target - train_output[1]) * self.activation(train_output[1])
        for i in range(self.hl + 1):
            self.how[i] += self.alpha * output_delta * train_output[1]

        hidden_delta = np.zeros(self.hl)
        for i in range(self.hl):
            hidden_delta[i] = self.how[i] * output_delta * self.activation(train_output[0][i])
        for i in range(self.il + 1):
            for j in range(self.hl):
                self.ihw[i] += self.alpha * hidden_delta[j] * train_output[0][j]

    def train(self, data, iteration=1000):
        for i in range(iteration):
            for datum in data:
                output = self.forward(datum[0])
                self.backward(output, datum[1][0])

    def test(self, data):
        for datum in data:
            print(datum[0], " -> ", self.forward(datum[0]))

    @staticmethod
    def activation(x):
        return x * (1 - x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))


def demo():
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    n = NN(2, 3, len(pat))
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)


if __name__ == '__main__':
    demo()
