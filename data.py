import glob
import skimage
import mnist_reader
from neural import sigmoid

class DataLoader:
  def __init__(self):
    self.trainingRatio = 0.7
    self.data = []
  def loadData(self):
    print("loading data training ratio is " + str(self.trainingRatio))
  def getNextLearningRecord(self):
    pass


class ImagesLoader(DataLoader):
  def convertToTable(self, imageDir):
    img = skimage.io.imread(imageDir)
    return skimage.img_to_float(img)
  
  def loadData(self):
    super().loadData()
    self.X_raw_train, self.y_train = mnist_reader.load_mnist("/home/user/python/fashion-mnist/data/fashion", kind="train")
    #self.X_raw_train = [[255,100,20,11,50], [223, 52, 12,255, 11]]
    #self.y_train = [1, 3]
    self.inputLen = len(self.X_raw_train[0])
    self.X_train = []
    for x in self.X_raw_train:
      new_column = []
      for y in x:
         new_column.append(y / 255)
      self.X_train.append(new_column)
    print("done")
#    self.X_test, self.y_test = mnist_reader.load_mnist("/home/user/python/fashion-mnist/data/fashion", kind="t10k")
