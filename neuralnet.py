#!/usr/bin/env python2
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
        for _0 in xrange(len(hidden_layer_sizes)):
            layer_neurons = []
            for _1 in xrange(layer_size_cycle.next()):
                # Initialize random weights for each of the inputs from
                # previous layer
                weights = [random.random()
                           for i in xrange(len(self.layers[-1])+1)]
                # Keep a tally of how many node weights there are overall
                self.num_node_weights += len(weights)
                layer_neurons.append(Neuron(weights, self.layers[-1]))
            self.layers.append(layer_neurons)

        # Set up the output layer
        output_layer=[]
        for _0 in xrange(n_outputs):
            # Initialize random weights for each of the inputs from prev. layer
            weights = [random.random() for i in xrange(len(self.layers[-1])+1)]
            self.num_node_weights += len(weights)
            output_layer.append(Neuron(weights, self.layers[-1]))

        # Add the output layer to the whole group of layers
        self.layers.append(output_layer)


    def GetOutput(self, input_values):
        assert len(input_values)==self.n_inputs

        # Set the input neuron values
        for i, neuron in enumerate(self.layers[0]):
            neuron.value = input_values[i]

        inputs = []
        for layer in self.layers:
            last_outputs = []
            for neuron in layer:
                last_outputs.append(neuron.GetValue(inputs))
            inputs = last_outputs

        return last_outputs


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


def test():
    net = NeuralNet(4, 2, [6, 7, 8, 9, 10])

    print len(net.layers)
    print map(len,net.layers)
    print(net.GetOutput((1,0.2,0.1,1)))
    weight=net.GetWeights()
    net.SetWeights(weight)
    print(net.GetOutput((1,2,0.5,1)))


# Profile the short test/demo
if __name__ == "__main__":
    import cProfile
    cProfile.run("test()")

