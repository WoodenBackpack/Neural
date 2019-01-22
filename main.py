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

class MainWindow(qWidgets.QMainWindow):
    def __init__(self, parent = None):
        qWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.neural = NeuralNetwork(2,3,2,2)
    def drawNet(self):
        circleSize = 50
        spacing = 20
        layerWithMaxSize = max(self.neural.layers, key=lambda x:x[0].size)
        maxLayerLen = layerWithMaxSize.size
        self.maxLen = maxLayerLen * (circleSize + spacing)
        startX = circleSize
        for layer in self.neural.layers:
            lenSize = layer[0].size * (circleSize + spacing)
            startPos = self.maxLen / 2 - (lenSize / 2)
            for cell in layer[0]:
                self.createAndDrawRect(startX, startPos, circleSize, circleSize, str(cell)[:4], self.painter, cell)
                startPos += circleSize + spacing
            startX += circleSize + spacing
    def keyPressEvent(self, event):
        if event.key() == qtCore.Qt.Key_Space:
            self.neural.process(0,0)
            self.neural.backPropagation(self.neural.generateExpectedTab(2, 0))
            self.update()
    def paintEvent(self, event):
        self.painter = qtGui.QPainter(self)
        self.painter.setPen(qtCore.Qt.white)
        self.painter.begin(self)
        self.drawNet();
        self.resize(500, self.maxLen)
        self.painter.end()
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
