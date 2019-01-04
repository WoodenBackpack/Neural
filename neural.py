
import numpy as np

class NeuralNetwork:
    def __init__(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen):
        self.layers = self.createAndRandomizeLayers(inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen)
    def createAndRandomizeLayers(self, inputLayerLen, hiddenLayersLen, hiddenLayers, outputLayersLen):
        inLayer = np.random.rand(1, inputLayerLen)
        layers = [inLayer]
        for x in range(hiddenLayers):
            layers.append(np.random.rand(1, hiddenLayersLen))
        outLayer = np.random.rand(1, outputLayersLen)
        layers.append(outLayer)
        return layers
