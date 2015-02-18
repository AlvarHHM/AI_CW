import numpy as np
import math
class NN:
    def __init__(self,inputLayer,hiddenLayer,outputLayer):
        self.il = inputLayer
        self.hl = hiddenLayer
        self.ol = outputLayer

        self.alpha = 0.1
        

    def forward(self,input,target):

        for i in range(self.hl):
            sum = 0
            #bias
            sum = sum + self.whl[0,i]
            for j in range(1,self.il+1):
                sum = sum + self.whl[j,i] * input[j]
            self.ho[i] = self.sigmoid(sum)

        for i in range(self.ol):
            sum = 0
            #bias
            sum = sum + self.wol[0,i]
            for j in range(1,self.ho+1):
                sum = sum + self.wol[j,i] * self.ho[j]
            self.oo[i] = self.sigmoid(sum)

    def backward(self,input,target):
        #only work when there is only one output
        o_delta = (target - self.oo[0]) * self.activation(self.oo[0])
        for j in range(self.hl+1):
            self.wol[j, 0] += self.alpha * o_delta * self.oo[0]

        for i in range(self.hl):
            h_delta = self.wol[i+1,0] *  o_delta * self.activation(self.ho[i])
            self.whl += self.alpha * h_delta * self.ho[i]
        
    def train(self,data,iteration=1000):
        self.whl = np.random.uniform(-1 * 2/data.length,2/len(data),[self.il+1,self.hl])
        self.wol = np.random.uniform(-1 * 2/data.length,2/len(data),[self.hl+1,self.ol])

        self.io = np.zeros(len(data))
        self.ho = np.zeros(self.hl)
        self.oo = np.zeros(self.ol)

        for i in range(iteration):
            for datum in range(len(data)):
                input = datum[0]
                target = datum[1]
                self.forward(input,target)

    def activation(x):
        return x * (1-x)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
