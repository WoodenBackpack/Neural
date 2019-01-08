import math
import numpy as np

class NeuralNetwork:
    def __init__(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen):
        self.layers = self.createAndRandomizeLayers(inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen)
        numberOfLayers = 2 + hiddenLayers
        maxLayerLen = max(inputLayerLen, hiddenLayersLen, outputLayersLen)
        self.weigths = np.random.rand(numberOfLayers, maxLayerLen, maxLayerLen)
        self.biases = np.random.randint(-5, 5, size = (numberOfLayers, maxLayerLen, maxLayerLen))
    def createAndRandomizeLayers(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen):
        inLayer = np.random.rand(1, inputLayerLen)
        layers = [inLayer]
        for x in range(hiddenLayers):
            layers.append(np.random.rand(1, hiddenLayersLen))
        outLayer = np.random.rand(1, outputLayersLen)
        layers.append(outLayer)
        return layers

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


    def process(self, layerIndex, cellIndex):
        if (layerIndex + 1 == len(self.layers)):
            return
        for x in range(self.layers[layerIndex + 1][0].size):
            value = self.layers[layerIndex + 1][0][x]
            value *= self.weigths[layerIndex, cellIndex, x] 
            value += self.biases[layerIndex, cellIndex, x]
            self.layers[layerIndex + 1][0][x] = self.sigmoid(value)
            self.process(layerIndex + 1, x)
    def generateExpectedTab(outputLength, pos):
        l = numpy.random.zeros(outputLength)
        l[0][pos] = 1
        return l
    def calculateCost(expectedOutput):
        cost = 0
        for (x, y) in (layers[:-1][0], expectedOutput):
            cost += (x - y) ** 2
        return cost

