import modelMaker
modelFile = open("trainedModels.txt", "r")

netStrings = []
for line in modelFile:

    stringNet = line.rstrip()
    netStrings.append(stringNet)

anns = modelMaker.Models(netStrings, 1)

testArray = []
testFile = open("testSet.txt", "r")

for line in testFile:

    line = line.rstrip()
    splitLine = line.split(", ")
    inputRow = [float(val) for val in splitLine]
    testArray.append(inputRow)

testFile.close()

anns.test(testArray)
