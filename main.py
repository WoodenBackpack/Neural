import PyQt5.QtCore as qtCore
import PyQt5.QtGui as qtGui
import PyQt5.QtWidgets as qWidgets

from neural import *
import math
import time
import data

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(500, 500)
        self.centralwidget = qWidgets.QWidget(MainWindow)
        self.horizontalLayout = qWidgets.QHBoxLayout(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        qtCore.QMetaObject.connectSlotsByName(MainWindow)

class MainWindow(qWidgets.QMainWindow):
    def __init__(self, parent = None):
        qWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.data = data.ImagesLoader()
        self.data.loadData()
        self.neural = NeuralNetwork(self.data.inputLen, 10 , 2, 10)
        self.it = 0
        print("started processing")
        processingStart = time.time()
        dataSize = len(self.data.X_train) - 1
        tenthPartOfDataSize = int(dataSize / 10)
        printed = 1
        for it in range(dataSize):
            self.neural.process(self.data.X_train[it], self.neural.generateExpectedTab(10, self.data.y_train[it]))
#        self.update()
            if it % tenthPartOfDataSize == 0:
                print("#" * printed)
                printed += 1
            if it % 25 == 0:
                self.neural.backwardPass()
        print("processed!")
        print("it taken: ", time.time() - processingStart)

    def drawNet(self):
        circleSize = 50
        spacing = 20
        layerWithMaxSize = max(self.neural.layers, key=lambda x:x[0].size)
        maxLayerLen = layerWithMaxSize.size
        self.maxLen = maxLayerLen * (circleSize + spacing)
        startX = circleSize
        lenSize = self.neural.layers[0].size * (circleSize + spacing)
        startPos = self.maxLen / 2 - (lenSize / 2)
        for inCell in range(self.neural.layers[0].size):
            self.createAndDrawRect(startX, startPos, circleSize, circleSize, str(self.neural.layers[0][inCell])[:4], self.painter, self.neural.layers[0][inCell])
            startPos += circleSize + spacing
        startX += circleSize + spacing
        for layer in range(1, len(self.neural.layers)):
            lenSize = self.neural.layers[layer][0].size * (circleSize + spacing)
            startPos = self.maxLen / 2 - (lenSize / 2)
            for cell in range(len(self.neural.layers[layer][1])):
                self.createAndDrawRect(startX, startPos, circleSize, circleSize, str(self.neural.layers[layer][1][cell])[:4], self.painter,self.neural.layers[layer][1][cell])
                startPos += circleSize + spacing
            startX += circleSize + spacing
    def keyPressEvent(self, event):
        if event.key() == qtCore.Qt.Key_Escape:
            self.close()
    def paintEvent(self, event):
        pass
#        self.painter = qtGui.QPainter(self)
#        self.painter.setPen(qtCore.Qt.white)
#        self.painter.begin(self)
#        self.drawNet();
#        self.resize(500, self.maxLen)
#        self.painter.end()
    def createAndDrawRect(self, xPos, yPos, x, y, text, painter, val):
        color = qtGui.QColor(0, 0, 255 - int(255 * val))
        qRect = qtCore.QRect(xPos ,yPos ,x ,y)
        qCircleRect = qtCore.QRect(xPos, yPos, x ,y)
        circleSize = 50
        center = qtCore.QPoint(xPos + 10, yPos + 25)
        self.painter.setBrush(color)
        self.painter.drawEllipse(qRect)
        self.painter.setBrush(qtCore.Qt.white)
        self.painter.drawText(center, text)

if __name__ == "__main__":
    app = qWidgets.QApplication(["app"])
    mainScreen = MainWindow()
    mainScreen.show()
    app.exec()
