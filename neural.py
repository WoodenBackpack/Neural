import math
import numpy as np

class Cell:
  def __init__(self, value = 0):
    self.value = value
  def getValue(self):
    return value

class HiddenCell(Cell):
  def __init__(self, value, out):
    super().__init__(value)
    self.out = out
  def getOutValue(self):
    return self.out

class Layer:
  def __init__(self, size):
    self.cells = []
    self.size = size

class InOutLayer(Layer):
  def __init__(self, size):
    super().__init__(size)
    for it in range(self.size):
      self.cells.append(Cell())

class HiddenLayer(Layer):
  def __init__(self, size):
    super().__init__(size)
    for it in range(self.size):
      self.cells.append(HiddenCell())

class NeuralNetwork:
    def __init__(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayerLen):
        self.hiddenLayersLen = hiddenLayersLen
        self.outputLayerLen = outputLayerLen
        self.layers = self.createAndRandomizeLayers(inputLayerLen, hiddenLayers)
        numberOfLayers = 2 + hiddenLayers
        maxLayerLen = max(inputLayerLen, hiddenLayersLen, self.outputLayerLen)
        self.weigths = np.random.rand(numberOfLayers - 1, maxLayerLen, maxLayerLen)
        self.biases = np.random.rand(numberOfLayers - 1, maxLayerLen)
        self.learningRate = 0.1
    def createAndRandomizeLayers(self, inputLayerLen, hiddenLayers):
        inLayer = np.random.rand(inputLayerLen)
        layers = [inLayer]
        for x in range(hiddenLayers):
            hiddenLayer = np.random.rand(2, self.hiddenLayersLen)
            layers.append(hiddenLayer)
        outLayer = np.random.rand(2, self.outputLayerLen)
        layers.append(outLayer)
        return layers

    def forwardPass(self):
        for hiddenLayerIt in range(1, len(self.layers) - 1):
            cellRange = self.layers[hiddenLayerIt][0]
            for cellIt in range(len(cellRange) - 1):
                value = 0
                if (hiddenLayerIt == 1):
                    previousLayerSize = len(self.layers[0])
                else:
                    previousLayerSize = len(self.layers[hiddenLayerIt - 1][0])
                for previousLayerIt in range(previousLayerSize - 1):
                    if (hiddenLayerIt == 1):
                        value += self.layers[hiddenLayerIt - 1][previousLayerIt] * self.weigths[hiddenLayerIt - 1][previousLayerIt][cellIt]
                    else:
                        value += self.layers[hiddenLayerIt - 1][1][previousLayerIt] * self.weigths[hiddenLayerIt - 1][previousLayerIt][cellIt]
                value += self.biases[hiddenLayerIt - 1][cellIt] * 1
                self.layers[hiddenLayerIt][0][cellIt] = value
                sigm = value / 255
                if round(sigm, 2) == 0.0:
                    sigm = 0.0
                self.layers[hiddenLayerIt][1][cellIt] = sigm
        for outputLayerIt in range(len(self.layers[-1][0])):
            value = 0
            for cellIt in range(len(self.layers[-2][0])):
                value += self.layers[-2][0][cellIt] * self.weigths[-2][cellIt][outputLayerIt]
            value += self.biases[-1][outputLayerIt] * 1
            self.layers[-1][0][outputLayerIt] = value
            sigm = sigmoid(value)
            if round(sigm, 3) == 0.0:
                sigm = 0.0
            if round(sigm, 3) == 0.0:
                sigm = 0.0
            self.layers[-1][1][outputLayerIt] = sigm
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
            for weigthIt in range(len(self.layers[-2])):
                weigth = self.weigths[-2][weigthIt][cellIt]
                totalToOut = 1 if out == 0.0 else total/out
                outToNet = 1 if net == 0.0 else out/net
                netToWeigth = 1 if weigth == 0.0 else net/weigth
                totalToWeigth = totalToOut * outToNet * netToWeigth
                self.weigths[-2][weigthIt][cellIt] -= self.learningRate * totalToWeigth

    def backwardPassForHiddenLayers(self):
        total = self.totalError
        for hiddenLayerIt in range(1, len(self.layers) - 1):
            for cellIt in range(len(self.layers[hiddenLayerIt][0])):
                if (hiddenLayerIt == 1):
                     out = self.layers[hiddenLayerIt][0][cellIt]            
                     net = self.layers[hiddenLayerIt][0][cellIt]
                else:
                     out = self.layers[hiddenLayerIt][1][cellIt]            
                     net = self.layers[hiddenLayerIt][0][cellIt]
                maxIter = self.layers[hiddenLayerIt - 1][0] if hiddenLayerIt != 1 else self.layers[hiddenLayerIt - 1]
                for weigthIt in range(len(maxIter)):
                    weigth = self.weigths[hiddenLayerIt - 1][weigthIt][cellIt]
                    totalToOut = 1 if out == 0.0 else total/out
                    outToNet = 1 if net == 0.0 else out/net
                    netToWeigth = 1 if weigth == 0.0 else net/weigth
                    totalToWeigth = totalToOut * outToNet * netToWeigth
                    self.weigths[hiddenLayerIt - 1][weigthIt][cellIt] -= self.learningRate * totalToWeigth

    def backwardPass(self):
        self.backwardPassForOutputLayer()
        self.backwardPassForHiddenLayers()

    def process(self, inputLayer, expected):
        self.layers[0] = inputLayer
        self.forwardPass()
        self.calculateTotalError(expected)

    def calculateCost(self, expectedVal, outputCellVal):
        cost = 0
        self.error.append(expectedVal - outputCellVal)
        self.totalError = sum(self.error)
    def generateExpectedTab(self, outputLength, pos):
        l = np.zeros(outputLength)
        l[pos] = 1
        return l

def sigmoid(x):
    try:
        sigm = 1 / (1 + math.exp(-x))
        if round(sigm, 3) == 0.0:
            sigm = 0.0
        return sigm
    except OverflowError:
        return 0.0
