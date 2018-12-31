
import numpy as np

class NeuralNetwork:
    class Node:
        def __init__(self):
            self.val = np.random.uniform(0, 1)
            print(self.val)


    def __init__(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen):
        self.node = self.Node()
        self.inLen = np.random.rand(1, inputLayerLen)
        self.hiddenLayers = []
        for x in range(hiddenLayers):
            self.hiddenLayers.append(np.random.rand(1, hiddenLayersLen))
        self.outLen = np.random.rand(1, outputLayersLen)

