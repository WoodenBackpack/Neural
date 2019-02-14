import glob
import skimage

class DataLoader:
  def __init__(self):
    self.trainingRatio = 0.7
    self.data = []
  def loadData(self):
    
  def getNextLearningRecord(self):
    


class ImagesLoader(DataLoader):
  def __init__(self, directory):
    files = glob.glob(directory + ".*\.png")
    self.data = [self.convertoToTable(image )for image in files]

  def convertToTable(self, imageDir):
    img = skimage.io.imread(imageDir)
    return skimage.img_to_float(img)
    
