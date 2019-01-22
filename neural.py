import math
import numpy as np

class NeuralNetwork:
    def __init__(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen):
        self.layers = self.createAndRandomizeLayers(inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen)
        numberOfLayers = 2 + hiddenLayers
        maxLayerLen = max(inputLayerLen, hiddenLayersLen, outputLayersLen)
        self.weigths = np.random.rand(numberOfLayers, maxLayerLen, maxLayerLen)
        self.biases = np.random.randint(-5, 5, size = (numberOfLayers, maxLayerLen))
        self.errors = np.zeros((numberOfLayers, maxLayerLen))
    def createAndRandomizeLayers(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen):
        inLayer = np.random.rand(1, inputLayerLen)
        layers = [inLayer]
        for x in range(hiddenLayers):
            layers.append(np.random.rand(1, hiddenLayersLen))
        outLayer = np.random.rand(1, outputLayersLen)
        self.outLayer = [self.sigmoid(x) for x in outLayer[0]]
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
            value += self.biases[layerIndex, cellIndex]
            self.layers[layerIndex + 1][0][x] = self.sigmoid(value)
            self.process(layerIndex + 1, x)

    def calculateCost(self, expectedVal, outputCellVal):
        cost = 0
        self.error.append(expectedVal - outputCellVal)
        self.totalError = sum(self.error)

    def backPropagation(self, expectedOutput):
        self.science = 0.5
        self.calculateForOutputLayer(expectedOutput)
        self.calculateForOtherLayers()
    def generateExpectedTab(self, outputLength, pos):
         l = np.zeros(outputLength)
         l[pos] = 1
         return l

    def calculateForOutputLayer(self, expectedOutput):
         lastLayerIndex = len(self.layers) - 1
         self.error = []
         for outputCell in range(self.layers[lastLayerIndex][0].size):
             self.calculateCost(expectedOutput[outputCell], self.layers[lastLayerIndex][0][outputCell])

    def calculateForOtherLayers(self):
        lastLayerIndex = len(self.layers) - 1
        print(self.layers)
        for layerIt in range(len(self.layers[lastLayerIndex][0]), 0, -1):
            for cellIt in range(len(self.layers[layerIt][0])):
                for weigthIt in range(len(self.layers[layerIt + 1][0])):
                    change = 0
                    for hiddenLayerIt in range(self.layers[lastLayerIndex][0].size):
                         print(str(layerIt) + " " + str(cellIt) + " " + str(weigthIt))
                         totalToOut = (self.totalError / self.layers[lastLayerIndex][0][hiddenLayerIt])
                         print(totalToOut)
                         outToNet = (self.sigmoid(self.layers[layerIt][0][cellIt]) / self.layers[layerIt][0][cellIt])
                         netToOuth = (self.layers[layerIt][0][cellIt] / self.weigths[layerIt][cellIt][weigthIt])
                         change += (totalToOut * outToNet * netToOuth)
                    print("before: " + str(self.weigths[layerIt][cellIt][weigthIt]))
                    self.weigths[layerIt][cellIt][weigthIt] -= (self.science * change)
                    print("after: " + str(self.weigths[layerIt][cellIt][weigthIt]))



#     def propagateBack(self, expectedOutput):
#         for layerIndex in range(len(self.layers)):
#             for cellIndex in range(self.layers[layerIndex][0].size):
#                 self.calculateCellInput(layerIndex, cellIndex)
#         lastLayerIndex = len(self.layers)
#         totalLayerError = 0;
#         for outputCellIndex in len(self.layers[lastLayerIndex]):
#             cellError = 1/2 * (expectedOutput - layers[lastLayerIndex]) ** 2
#             totalLayerError += cellError
#         self.calculateCellInput(lastLayerIndex, 0)
# 
#     def calculateCellInput(self, layerIndex, cellIndex):
#         print("processing " + str(layerIndex) + " " + str(cellIndex) )
#         cellInput = 0
#         currentCellIndex = cellIndex - 1
#         currentLayerIndex = layerIndex - 1
#         if (layerIndex == 0 or cellIndex == 0):
#             return
#         for prevCell in range(self.layers[currentLayerIndex][currentCellIndex].size):
#             cellInput += self.layers[layerIndex - 1][prevCell] * self.weigths[layerIndex, cellIndex, prevCell]
#             cellInput += self.biases[layerIndex, cellIndex]
#             self.calculateCellInput(currentLayerIndex, currentCellIndex)
#             return (1 / 1 + math.e**(-cellInput))
#         
