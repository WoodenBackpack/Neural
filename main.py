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


        neural = NeuralNetwork(10, 3, 4, 2)
        startPos = 10
        for x in neural.inLen[0]:
            self.createAndDrawRect(10, startPos, 30, 30, str(x)[:4], self.painter)
            startPos += 35
        startX = 50
        for x in neural.hiddenLayers:
            startPos = 10
            for y in x[0]:
                self.createAndDrawRect(startX, startPos, 30, 30, str(y)[:4], self.painter)
                startPos += 35
            startX += 40

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
