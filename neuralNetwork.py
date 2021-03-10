import random
import math

class AnnModel:

    def __init__(self, inputs, learning):

        """Creates an ANN, by initializing the number of hidden nodes in the hidden layer and then assigning random values to
            all biases and connection weights"""

        # number of hidden nodes, learning parameter
        self.numHidden = random.randint(int(inputs/2), (2*inputs))
        self.startLearning = learning
        self.stopLearning = learning/10
        self.learning = learning
        self.alpha = 0.9

        """Neuron dictionary, where 
        key:value 
        neuron number:[neuron bias, [neuron input connection weights]]
        """
        self.neurons = {}

        biasRange = 2/inputs
        self.weightChanges = []

        # For loop to create all nodes, number of hidden nodes, plus one output node
        for i in range(self.numHidden + 1):

            # Array to be filled with 0s which will hold previous weight changes of neuron connection weights/bias
            # in order to apply momentum to the training of MLP
            prevChanges = []

            # If output node then its bias initialization range should be based on number of hidden nodes
            if i == self.numHidden:
                biasRange = 2/self.numHidden
                inputs = self.numHidden

            # Creates a random bias for neuron within bias initialization range
            self.neurons[i] = []
            neuronBias = round(random.uniform(-biasRange, biasRange), 2)
            self.neurons[i].append(neuronBias)
            prevChanges.append(0)

            """ Creates list of weights for connections leading into node and then adds a randomised weight created
            within weight initialization range for each connection"""
            weights = []
            for j in range(inputs):

                # Weight for each connection leading into the neuron is randomised
                conWeight = round(random.uniform(-biasRange, biasRange), 2)
                weights.append(conWeight)
                prevChanges.append(0)

            # Adds connection weights to neuron in neuron dictionary
            self.neurons[i].append(weights)
            # Adds previous weight changes for this node to array holding previous weight changes of all node
            self.weightChanges.append(prevChanges)

    """Updates the weights/biases of a neuron after a forward pass has been executed and the delta for that neuron has
     been calculated using the derivative of the sigmoid function. 
     inputVals should be an array of all the values being passed in to the selected neuron"""
    def updateNeuron(self, neuron, delta, inputVals, neuronKey):

        prevChanges = self.weightChanges[neuronKey]
        # Neuron bias is updated by adding the (learning parameter x delta) to its previous weight
        neuron[0] += ((self.learning * delta) + (prevChanges[0]*self.alpha))
        neuron[0] = round(neuron[0], 8)
        weights = neuron[1]

        # Iterates through every weight of connections leading into node and updates them by adding (learning parameter
        # x delta x value of node on other end of connection) to its original weight
        for x in range(len(weights)):
            weight = weights[x]
            inputVal = inputVals[x]
            updatedWeight = weight + (self.learning*delta*inputVal) + (self.alpha * prevChanges[x+1])
            prevChanges[x+1] = updatedWeight - weight
            neuron[1][x] = round(updatedWeight, 8)

        self.weightChanges[neuronKey] = prevChanges

    """Function to carry out simulated annealing, as the number of epochs increased, the learning parameter decreases"""
    def adjustLearning(self, epochs, maxEpochs):

        self.learning = self.stopLearning + ((self.startLearning - self.stopLearning) * (1- (1/(1+math.e**(10-(20*epochs/maxEpochs))))))

    """Function to carry out a forward pass on the ANN, has one parameter which is an array holding the input values 
    being passed on to the hidden layer"""
    def forwardPass(self, inputVals):

        # hiddenVals stores all the sigmoid values of the hidden nodes in this forward pass, the last item in the list
        # is the model's output
        hiddenVals = []
        outputNode = self.numHidden

        # Iterator through every node in the model
        for node in range(self.numHidden+1):

            neuronSum = 0
            weights = (self.neurons[node])[1]

            """For every connection leading into the neuron, the connection weight multiplied by the connection's input
            value is added to the neuron's sum total"""
            for x in range(len(weights)):

                if node != (outputNode):
                    neuronSum += (weights[x]*inputVals[x])

                else:
                    neuronSum += (weights[x]*hiddenVals[x])

            # Adds bias to neuron sum
            neuronSum += self.neurons[node][0]

            # Applies Sigmoid function to the neuron sum to find u
            funcS = 1/(1 + (math.e ** -neuronSum))

            # Adds the sigmoid value of neuron to the hiddenVals array
            hiddenVals.append(round(funcS, 8))

        return hiddenVals

    """Function to carry our a backward pass on the ANN
    sArray is the array holding all sigmoid values of the model's neurons from a forward pass
    correct is the correct output value
    inputVals is an array holding the original input values to the model fo the forward pass"""
    def backwardPass(self, sArray, correct, inputVals):

        outputNode = self.numHidden
        output = sArray[outputNode]
        outputWeights = self.neurons[outputNode][1]
        deltas = []

        # Iterator through all nodes beginning at the output node
        for node in range(self.numHidden, -1, -1):

            # Finds the derivative of the sigmoid function for the node
            derivFuncS = (sArray[node]*(1-sArray[node]))

            # If the node is the output node it finds its delta value by finding the difference between the correct
            # output and the model's output
            if node == outputNode:
                deltaOutput = round((correct-output)*derivFuncS, 8)
                deltas.append(deltaOutput)

            else:
                # Rounds delta value for node to 4 decimal places and inserts it at front of deltas array to keep them
                # in correct node order
                delta = round(outputWeights[node]*deltaOutput*derivFuncS, 8)
                deltas.insert(0, delta)

        # Updates weights and biases of all hidden nodes
        for i in range(self.numHidden):
            self.updateNeuron(self.neurons[i], deltas[i], inputVals, i)

        # Updates weights and biases of output neuron
        self.updateNeuron(self.neurons[outputNode], deltas[outputNode], sArray, self.numHidden)
