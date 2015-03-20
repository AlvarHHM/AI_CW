from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import *
import csv
import math
import random


def readDataFromFile(file):
    data = []
    with open(file, 'rbU') as csvfile:
        spamreader = csv.reader(csvfile, dialect=csv.excel_tab)
        for row in spamreader:
            row_data = [float(x) for x in row[0].split(',')]
            data.append([row_data[:-1], [row_data[-1]]])
    return data


def main():
    net = buildNetwork(8, 3, 1, bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)
    ds = SupervisedDataSet(8, 1)
    data = readDataFromFile('CWDatav6.csv')
    random.shuffle(data);
    train_set = data[int(0.2*len(data)):]
    test_set = data[:int(0.2*len(data))]
    [ds.addSample(x[0], x[1]) for x in train_set]
    trainer = BackpropTrainer(net, ds, learningrate=0.1, momentum=0.9, verbose=True)

    # for inp, tar in ds:
    #     print [net.activate(inp), tar]

    trainer.trainUntilConvergence()
    # for i in range(1, 1000):
    #     print 'error ' + str(trainer.train())

    error = 0
    for inp, tar in test_set:
        print net.activate(inp), tar
        error += (tar - net.activate(inp)) ** 2
    print math.sqrt(error/len(test_set))

if __name__ == '__main__':
    main()