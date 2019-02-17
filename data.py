import glob
import mnist_reader
import time
from neural import sigmoid

class ImagesLoader:
  def loadData(self):
    print("started loading data")
    startTime = time.time()
    self.X_raw_train, self.y_train = mnist_reader.load_mnist("/home/user/git-repos/fashion-mnist/data/fashion", kind="train")
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
    print("it took: ", time.time() - startTime)
#    self.X_test, self.y_test = mnist_reader.load_mnist("/home/user/python/fashion-mnist/data/fashion", kind="t10k")
