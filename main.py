import modelMaker

"""For each data point in the training set the line is stripped of all extra white space, and split so that the 
    input values are stored in an array. List comprehension is used to change all array values to floats"""
trainingArray = []
training = open("trainingSet.txt", "r")

for line in training:
    line = line.rstrip()
    splitLine = line.split(", ")
    inputRow = [float(val) for val in splitLine]
    trainingArray.append(inputRow)

training.close()

"""For each data point in the validation set the line is stripped of all extra white space, and split so that the 
    input values are stored in an array. List comprehension is used to change all array values to floats"""
validationArray = []
validation = open("validationSet.txt", "r")

for line in validation:

    line = line.rstrip()
    splitLine = line.split(", ")
    inputRow = [float(val) for val in splitLine]
    validationArray.append(inputRow)

validation.close()


# Opens and reads inputs and correct output from data file containing test data points
testArray = []
testFile = open("testSet.txt", "r")

for line in testFile:

    line = line.rstrip()
    splitLine = line.split(", ")
    inputRow = [float(val) for val in splitLine]
    testArray.append(inputRow)

testFile.close()

# Sets the maximum number of epochs a network will be trained for and the number of networks to be created
epochs = int(input("Max number of epochs? "))
numNetworks = int(input("How many neural networks would you like to train? "))
# Creates an array of neural networks
models = (modelMaker.Models(numNetworks))
# Trains the array of neural networks
models.train(epochs, trainingArray, validationArray)
models.test(testArray)
