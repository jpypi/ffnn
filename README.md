# Python Feed Forward Neural Net

Just a simple frame for setting up a feed-forward neural net with weights. This
is partially just so, having built the framework I obtain a better understanding
of the inner workings of neural nets.

Note: This does not proivde any built-in backpropagation. This is only the
neurons and net.

To get values from the net simply call GetOutput() with values for the input
layer and GetValue() will be called successivly from the output layer to its
parents and to that layer's parents etc. propgating all the way to the input
neurons.

## Todo

- Potentially try to cache results of GetValue at each neuron to reduce the
  inefficiency of calling GetValue all the way up to the input for every node.
  (Or maybe make value propogate forward, though potentially a little more
  complex, this would likely be more efficient.)


Ok, enough writing here. On to using this thing with a GA!
