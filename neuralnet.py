import random
import itertools

# Other NN classes
from neuron import Neuron
import helpers


class NeuralNet:
    def __init__(self, n_inputs, n_outputs, hidden_layer_sizes=[]):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layers = []
        self.num_node_weights = 0

        # Set up the input layer
        self.layers.append([Neuron() for _ in xrange(n_inputs)])

        # Generate the hidden layers
        layer_size_cycle = itertools.cycle(hidden_layer_sizes)
        for _0 in xrange(len(hidden_layer_sizes):
            layer_neurons = []
            for _1 in xrange(layer_size_cycle.next()):
                # Initialize random weights for each of the inputs from
                # previous layer
                weights = [random.random() for i in xrange(len(self.layers[-1]))]
                # Keep a tally of how many node weights there are overall
                self.num_node_weights += len(weights)
                layer_neurons.append(Neuron(weights, self.layers[-1]))
            self.layers.append(layer_neurons)

        # Set up the output layer
        output_layer=[]
        for _0 in xrange(n_outputs):
            # Initialize random weights for each of the inputs from prev. layer
            weights = [random.random() for i in xrange(len(self.layers[-1]))]
            self.num_node_weights += len(weights)
            output_layer.append(Neuron(weights, self.layers[-1]))

        # Add the output layer to the whole group of layers
        self.layers.append(output_layer)


    def GetOutput(self, input_values):
        """
        Calling this will cause a .GetValue() call to propgate backwards
        through every node in the net starting at the output nodes. This action
        will result in getting the values of the output layer nodes
        len(input_values) must == len(n_inputs) set at initialization of net
        """

        assert len(input_values)==self.n_inputs

        input_values = helpers.scale_values(input_values)
        # Set the input neuron values
        for i, neuron in enumerate(self.layers[0]):
            neuron.value = input_values[i]

        return [neuron.GetValue() for neuron in self.layers[-1]]


    def GetWeights(self):
        """
        Returns all of the neuron weights in one flat list.
        Starts with left-most layer and top-most node.
        """

        weights = []
        for layer in self.layers:
            for neuron in layer:
                weights.extend(neuron.weights)

        return weights


    def SetWeights(self, weights):
        """
        Set the weights of all the neurons in one fell swoop
        """
        assert len(weights) == self.num_node_weights
        weights = iter(weights)
        for layer in self.layers:
            for neuron in layer:
                neuron.weights=[weights.next() for _ in neuron.weights]


if __name__ == "__main__":
    net = NeuralNet(4, 2, 4, [6])

    print len(net.layers)
    print map(len,net.layers)
    print(net.GetOutput((1,2,0.5,1)))
    weight=net.GetWeights()
    net.SetWeights(weight)
    print(net.GetOutput((1,2,0.5)))

