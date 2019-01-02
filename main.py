#!./bin/python3

import PyQt5.QtCore as qtCore
import PyQt5.QtGui as qtGui
import PyQt5.QtWidgets as qWidgets

from neural import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(500, 500)
        self.centralwidget = qWidgets.QWidget(MainWindow)
        self.horizontalLayout = qWidgets.QHBoxLayout(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        qtCore.QMetaObject.connectSlotsByName(MainWindow)

class MyScreen(qWidgets.QMainWindow):
    def __init__(self, parent = None):
        qWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
    def paintEvent(self, event):
        self.painter = qtGui.QPainter(self)
        self.painter.setPen(qtCore.Qt.black)
        self.painter.begin(self)



        neural = NeuralNetwork(8, 16, 2, 4)
        maxLayerLen = max([neural.inLen[0].size, neural.hiddenLayers[0].size, neural.outLen[0].size])
        maxLen = maxLayerLen * 40
        lenSize = neural.inLen[0].size * 40
        startPos = maxLen / 2 - (lenSize / 2)
        for x in neural.inLen[0]:
            self.createAndDrawRect(10, startPos, 30, 30, str(x)[:4], self.painter)
            startPos += 40
        startX = 50
        for x in neural.hiddenLayers:
            lenSize = x[0].size * 40
            startPos = maxLen / 2 - (lenSize / 2)
            for y in x[0]:
                self.createAndDrawRect(startX, startPos, 30, 30, str(y)[:4], self.painter)
                startPos += 40
            startX += 40

        lenSize = neural.outLen[0].size * 40
        startPos = maxLen / 2 - (lenSize / 2)
        for x in neural.outLen[0]:
            self.createAndDrawRect(startX, startPos, 30, 30, str(x)[:4], self.painter)
            startPos += 40

        self.resize(500, maxLen)
        self.painter.end()
    def createAndDrawRect(self, xPos, yPos, x, y, text, painter):
        qRect = qtCore.QRect(xPos ,yPos ,x ,y)
        self.painter.drawText(qRect, 20, text)
        self.painter.drawRect(qRect)

if __name__ == "__main__":
    app = qWidgets.QApplication(["app"])
    mainScreen = MyScreen()
    mainScreen.show()
    app.exec()
