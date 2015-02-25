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
                sum = sum + self.whl[j,i] * input[j-1]
            self.ho[i] = self.sigmoid(sum)

        for i in range(self.ol):
            sum = 0
            #bias
            sum = sum + self.wol[0,i]
            for j in range(1,self.hl+1):
                sum = sum + self.wol[j,i] * self.ho[j-1]
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
        self.whl = np.random.uniform(-1 * 2/len(data),2/len(data),[self.il+1,self.hl])
        self.wol = np.random.uniform(-1 * 2/len(data),2/len(data),[self.hl+1,self.ol])

        self.io = np.zeros(len(data))
        self.ho = np.zeros(self.hl)
        self.oo = np.zeros(self.ol)

        for i in range(iteration):
            for datum in data:
                input = datum[0]
                target = datum[1]
                self.forward(input,target)
                self.backward(input,target)

    def activation(self,x):
        return x * (1-x)

    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))
def demo():
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    #n.test(pat)



if __name__ == '__main__':
    demo()
