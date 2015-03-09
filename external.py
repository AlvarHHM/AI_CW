from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import *
import csv


def readDataFromFile(file):
    data = []
    with open(file, 'rbU') as csvfile:
        spamreader = csv.reader(csvfile, dialect=csv.excel_tab)
        for row in spamreader:
            row_data = [float(x) for x in row[0].split(',')]
            data.append([row_data[:-1], [row_data[-1]]])
    return data


def main():
    net = buildNetwork(8, 5, 1, bias=True, hiddenclass=SigmoidLayer)
    ds = SupervisedDataSet(8, 1)
    data = readDataFromFile('TestData.csv')
    [ds.addSample(x[0], x[1]) for x in data]
    trainer = BackpropTrainer(net, ds)

    # for inp, tar in ds:
    #     print [net.activate(inp), tar]

    # trainer.trainUntilConvergence()
    for i in range(1, 10000):
        print 'error ' + str(trainer.train())

    # for inp, tar in ds:
    #     print [net.activate(inp), tar]

if __name__ == '__main__':
    main()