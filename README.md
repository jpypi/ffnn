# Python Feed Forward Neural Net

Just a simple frame for setting up a feed-forward neural net. This is partially
just so, having built the framework I have a better understanding of the inner
workings of neural nets.

*Note:* This _does not_ proivde any backpropagation. This is only the
neurons and net.

## Details
- Net weights can all easily be pulled and replaced simultaneously (useful for
  genetic algorithms.
- All layers simply use all the outputs from the previous layer as inputs.
- Supports an arbitrary number of hidden layers of arbitrary sizes.
- Initial state for all weights on all neurons is given by Python's
  random.random()

Ok, enough here. On to using this thing with a genetic algorithm.
