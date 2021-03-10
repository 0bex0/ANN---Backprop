import neuralNetwork

class Models:

    def __init__(self, numModels):

        self.nets = []
        self.trainedMLP = []
        self.numModels = numModels

        """Creates file holding info on all created neural networks
        Data is stored in format "number of hidden nodes - initial learning parameter - initial neuron biases/weights"""
        numInputs = int(input("How many inputs do you want the ANN to have? "))
        networkFile = open("neuralNetworks.txt", "w")
        learningPara = float(input("What step size would you like to initialize the ANN with? "))

        for i in range(self.numModels):

            """Creates 'numModels' number of neural networks, with the given number of inputs and the specified 
            learning parameter, then creates a string representation of said neural network and writes it to the 
            file containing all the neural networks info """
            ann = neuralNetwork.AnnModel(numInputs, learningPara)
            self.nets.append(ann)
            annString = f"{ann.numHidden} - {ann.learning} - {ann.neurons} \n"
            networkFile.write(annString)
        networkFile.close()

    """Writes final biases and weights of a an array of trained neural networks to a text file"""
    def writeTrainedModels(self):

        trainedFile = open("trainedModels.txt", "w")

        for network in self.trainedMLP:
            annString = f"{network.numHidden} - {network.learning} - {network.neurons} \n"
            trainedFile.write(annString)

        trainedFile.close()

    def train(self, epochs, training, validation):

        validationLen = len(validation)

        # Trains each network
        for ann in self.nets:

            prevMSE = 100

            # For loop iterating for maximum number of epochs
            for i in range(epochs):

                # if i%50 == 0:
                #     ann.adjustLearning(i, epochs)

                """last item of input data row is removed and stored as correct output value,
                then input is passed into forward pass"""
                for row in training:

                    correct = row.pop()
                    # Forward pass produces an array of sigmoid values for each node, this then gets passed into
                    # backwards pass to adjust weights accordingly
                    fPass = ann.forwardPass(row)
                    ann.backwardPass(fPass, correct, row)
                    row.append(correct)

                """After a set number of epochs this starts to check if the network is being over-trained, if the 
                MSE of the validation set begins to increase the network stops training"""
                if i > (epochs/2):

                    mse = 0

                    # For each data row in the validation set the (observed value - modelled value)^2 is added to mse
                    for validationRow in validation:

                        observed = validationRow.pop()
                        fPass = ann.forwardPass(validationRow)
                        modelled = fPass.pop()
                        mse += ((observed-modelled)**2)
                        validationRow.append(observed)

                    # MSE is calculated by dividing sum of all MSE values by the number of data rows
                    mse = mse/validationLen

                    # If the MSE has increased then the model breaks out of its training loop
                    if mse > prevMSE:
                        print("epochs:", i)
                        print(mse)
                        print(prevMSE)
                        break
                    # Otherwise, the MSE becomes previous MSE so it can be compared in next iteration
                    else:
                        prevMSE = mse

            self.trainedMLP.append(ann)

        self.writeTrainedModels()

    """Tests all MLPs in array against a test set"""
    def test(self, testArray):

        print("hit")
        testResults = open("annTestResults.txt", "w")

        # Iterates over every MLP and assesses the MSE of the test set using that MLP
        for net in self.trainedMLP:

            mse = 0
            testLen = len(testArray)

            # For each data row in the test set the MSE is calculated and added to overall MSE
            for testRow in testArray:

                observed = testRow.pop()
                fPass = net.forwardPass(testRow)
                modelled = fPass.pop()
                mse += ((observed-modelled)**2)
                testRow.append(observed)

            # Average MSE for test set is then calculated and printed
            mse = (mse/testLen)**(1/2)
            testResults.write(str(mse) + "\n")

        testResults.close()

