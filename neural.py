import math
import numpy as np

class NeuralNetwork:
    def __init__(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayerLen):
        self.hiddenLayersLen = hiddenLayersLen
        self.outputLayerLen = outputLayerLen
        self.layers = self.createAndRandomizeLayers(inputLayerLen, hiddenLayers)
        numberOfLayers = 2 + hiddenLayers
        maxLayerLen = max(inputLayerLen, hiddenLayersLen, self.outputLayerLen)
        self.weigths = np.random.rand(numberOfLayers - 1, maxLayerLen, maxLayerLen)
        self.biases = np.random.rand(numberOfLayers - 1, maxLayerLen)
        self.learningRate = 0.5
    def createAndRandomizeLayers(self, inputLayerLen, hiddenLayers):
        inLayer = np.random.rand(inputLayerLen)
        layers = [inLayer]
        for x in range(hiddenLayers):
            hiddenLayer = np.random.rand(2, self.hiddenLayersLen)
            layers.append(hiddenLayer)
        outLayer = np.random.rand(2, self.outputLayerLen)
        layers.append(outLayer)
        return layers

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def forwardPass(self):
        for hiddenLayerIt in range(1, len(self.layers[1:-1][0])):
            for cellIt in range(len(self.layers[hiddenLayerIt][0])):
                value = 0
                for previousLayerIt in range(self.layers[hiddenLayerIt - 1].size):
                    value += self.layers[hiddenLayerIt - 1][previousLayerIt] * self.weigths[hiddenLayerIt - 1][previousLayerIt][cellIt]
                value += self.biases[hiddenLayerIt - 1][cellIt] * 1
                self.layers[hiddenLayerIt][0][cellIt] = value
                self.layers[hiddenLayerIt][1][cellIt] = self.sigmoid(value)
        for outputLayerIt in range(len(self.layers[-1][0])):
            value = 0
            for cellIt in range(len(self.layers[-2][0])):
                value += self.layers[-2][0][cellIt] * self.weigths[-2][cellIt][outputLayerIt]
            value += self.biases[-1][outputLayerIt] * 1
            self.layers[-1][0][outputLayerIt] = value
            self.layers[-1][1][outputLayerIt] = self.sigmoid(value)
            print("calculate forward for " + str(outputLayerIt))
    def calculateTotalError(self, expectedTab):
        self.error = []
        for x in range(self.outputLayerLen):
            self.error.append(0.5 * ((expectedTab[x] - self.layers[-1:][0][1][x]) ** 2))
        self.totalError = sum(self.error)

    def backwardPassForOutputLayer(self):
        total = self.totalError
        for cellIt in range(self.outputLayerLen):
            out = self.layers[-1][1][cellIt]            
            net = self.layers[-1][0][cellIt]
            print("back for cell = " + str(cellIt))
            for weigthIt in range(len(self.layers[-2])):
                weigth = self.weigths[-2][weigthIt][cellIt]
                totalToWeigth = (total/out) * (out/net) * (net/weigth)
                self.weigths[-2][weigthIt][cellIt] -= self.learningRate * totalToWeigth

    def backwardPassForHiddenLayers(self):
        total = self.totalError
        for hiddenLayerIt in range(1, len(self.layers) - 1):
            for cellIt in range(len(self.layers[hiddenLayerIt][0])):
                if (hiddenLayerIt == 1):
                     out = self.layers[hiddenLayerIt][0][cellIt]            
                     net = self.layers[hiddenLayerIt][0][cellIt]
                else:
                     out = self.layers[hiddenLayerIt][0][1][cellIt]            
                     net = self.layers[hiddenLayerIt][0][0][cellIt]
                for weigthIt in range(len(self.layers[hiddenLayerIt - 1])):
                    weigth = self.weigths[hiddenLayerIt - 1][weigthIt][cellIt]
                    totalToWeigth = (total/out) * (out/net) * (net/weigth)
                    self.weigths[hiddenLayerIt - 1][weigthIt][cellIt] -= self.learningRate * totalToWeigth

    def backwardPass(self):
        self.backwardPassForOutputLayer()
        self.backwardPassForHiddenLayers()


    def process(self, expected):
        self.forwardPass()
        self.calculateTotalError(expected)
        self.backwardPass()

#    def process(self, layerIndex, cellIndex):
#        if (layerIndex + 1 == len(self.layers)):
#            return
#        for x in range(self.layers[layerIndex + 1][1].size):
#            value = self.layers[layerIndex + 1][1][x]
#            value *= self.weigths[layerIndex, cellIndex, x]
#            value += self.biases[layerIndex, cellIndex]
#            self.layers[layerIndex + 1][0][x] = value
#            self.layers[layerIndex + 1][1][x] = self.sigmoid(value)
#            self.process(layerIndex + 1, x)

    def calculateCost(self, expectedVal, outputCellVal):
        cost = 0
        self.error.append(expectedVal - outputCellVal)
        self.totalError = sum(self.error)
    def backPropagation(self, expectedOutput):
        self.science = 0.5

#        self.calculateForOutputLayer(expectedOutput)
#        self.calculateForOtherLayers()

    def generateExpectedTab(self, outputLength, pos):
        l = np.zeros(outputLength)
        l[pos] = 1
        return l

#    def calculateForOutputLayer(self, expectedOutput):
#        lastLayerIndex = len(self.layers) - 1
#        self.error = []
#        for outputCell in range(self.layers[lastLayerIndex][1].size):
#            self.calculateCost(expectedOutput[outputCell], self.layers[lastLayerIndex][1][outputCell])
#
#    def calculateForOtherLayers(self):
#        lastLayerIndex = len(self.layers) - 1
#        for layerIt in range(len(self.layers[lastLayerIndex][1]), 0, -1):
#            for cellIt in range(len(self.layers[layerIt][1])):
#                for weigthIt in range(len(self.layers[layerIt + 1][1])):
#                    change = 0
#                    for hiddenLayerIt in range(self.layers[lastLayerIndex][1].size):
#                         print("params:")
#                         print("layers[layerIt[1][cellIt]]" + str(self.layers[layerIt][1][cellIt]))
#                         x = (self.layers[layerIt][1][cellIt])
#                         out = math.log1p(x / 1 - x)
#                         print ("logp1 = " + str(out))
#                         totalToOut = (self.totalError / out)
#                         outToNet = (out / self.layers[layerIt][0][cellIt])
#                         netToOuth = (self.layers[layerIt][0][cellIt] / self.layers[layerIt][1][cellIt])
#                         change += (totalToOut * outToNet * netToOuth)
#                    print("before: " + str(self.weigths[layerIt][cellIt][weigthIt]))
#                    self.weigths[layerIt][cellIt][weigthIt] -= (self.science * change)
#                    print("after: " + str(self.weigths[layerIt][cellIt][weigthIt]))
#
#

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
