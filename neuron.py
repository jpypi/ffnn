#!/usr/bin/env python2
from . import helpers


class Neuron:
    def __init__(self, weights=[], inputs=[]):
        self.weights = weights
        self.inputs  = inputs
        self.value   = None
        self.bias    = -1

    def GetValue(self, values = []):
        """
        When setting Neuron.value it would be wise to scale
        the value to between 0 and 1 so that the sigmoid
        function can do its thing better
        """

        # This "if" is used so that Neurons can be used for the input
        # layer and input can be set by simply setting the value
        if self.value != None:
            return self.value # helpers.sigmoid(self.value)

        bias_weight_index = len(self.weights) - 1


        # Find the sum of weighted inputs
        input_sum = 0
        # Use passed in values if given
        if values:
            # This won't work if the weights and inputs
            # lists are different lengths
            assert len(self.weights) - 1 == len(values)
            for i, value in enumerate(values):
                if i < bias_weight_index:
                    input_sum += value * self.weights[i]
        else:
            # This won't work if the weights and inputs
            # lists are different lengths
            assert len(self.weights) - 1 == len(self.inputs)

            for i, neuron in enumerate(self.inputs):
                if i < bias_weight_index:
                    input_sum += neuron.GetValue() * self.weights[i]

        input_sum += self.weights[bias_weight_index] * self.bias

        return helpers.sigmoid(input_sum)


# Short "unit" test
if __name__ == "__main__":
    n1 = Neuron()
    n1.value = 2
    n2 = Neuron()
    n2.value = 3

    n3 = Neuron([0.5, 0.25, 1], [n1, n2])
    print(n3.GetValue())
    print(n3.GetValue() == helpers.sigmoid(2*0.5+3*0.25+-1*1))
