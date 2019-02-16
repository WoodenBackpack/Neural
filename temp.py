import mnist_reader

X_raw_train, y_train = mnist_reader.load_mnist("/home/user/python/fashion-mnist/data/fashion", kind="train")

with open("X_train.txt", "w") as outFile:
  for cell in X_raw_train:
    for value in cell[:-1]:
      outFile.write(str(round(value / 255, 3) + ";"))
    outFile.write(str(round(value / 255, 3)))
    outFile.write("\n")

with open("y_train.txt", "w") as outFile:
  for cell in y_train:
    for value in cell[:-1]:
      outFile.write(str(round(value / 255, 3) + ";"))
    outFile.write(str(round(value / 255, 3)))
    outFile.write("\n")
