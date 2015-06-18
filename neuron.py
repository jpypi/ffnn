import helpers


class Neuron:
    def __init__(self, weights=[], inputs=[]):
        self.weights = weights
        self.inputs  = inputs
        self.value   = None

    def GetValue(self):
        """
        When setting Neuron.value it would be wise to scale
        the value to between 0 and 1 so that the sigmoid
        function can do its thing better
        """

        # This "if" is used so that Neurons can be used for the input
        # layer and input can be set by simply setting the value
        if self.value != None:
            return helpers.sigmoid(self.value)

        # This won't work if the weights and inputs
        # lists are different lengths
        assert len(self.weights) == len(self.inputs)

        # Find the sum of weighted inputs
        input_sum = 0
        for i, neuron in enumerate(self.inputs):
            input_sum += neuron.GetValue() * self.weights[i]

        return helpers.sigmoid(input_sum)


if __name__ == "__main__":
    n1 = Neuron()
    n1.value = 2
    n2 = Neuron()
    n2.value = 3

    n3 = Neuron([0.5, 0.25], [n1, n2])
    print(n3.GetValue())
    print(n3.GetValue() == helpers.sigmoid(2*0.5+3*0.25))
