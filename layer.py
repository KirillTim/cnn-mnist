import numpy as np
from scipy.special import expit
from scipy.signal import convolve2d

"""
Base class for a neural network layer
"""
class NetworkLayer(object):
    """
    Initializes a new instance of NetworkLayer.
    Parameters:
        * shape - shape of the layer output
    """
    def __init__(self, shape: tuple):
        self.shape = shape

    """
    Set data for an input layer of the network.
    Parameters:
        * input: new data for the input layer
    """
    def set_input(self, input):
        self.prev.set_input(input)

    """
    Perform a forward propagation pass from the initial layer to the current.
    """
    def forward(self):
        pass

    """
    Perform a back propagation pass from the current layer to the initial.
    Parameters:
        * next_deriv: loss function derivative for outputs of this layer
        * learn_rate: learning rate of the pass
    """
    def backward(self, next_deriv: np.ndarray, learn_rate: float):
        pass

"""
Represents an input layer of a neural network.
"""
class InitialLayer(NetworkLayer):
    def __init__(self, shape):
        super(InitialLayer, self).__init__(shape)
    
    def set_input(self, input):
        self.output = input

"""
Represents a full layer for a neural network.
"""
class FullLayer(NetworkLayer):
    """
    Initializes a new instance of FullLayer.
    Parameters:
        * prev - previous layer of the network (a layer that produces results which are further processed by this layer)
        * size - output shape of this layer
        * activation - activation function of this layer ('sigmoid' and 'linear' are supported at the moment)
    """
    def __init__(self, prev: NetworkLayer, size: tuple, activation='sigmoid'):
        super(FullLayer, self).__init__(size)
        #self.neurons = np.random.normal(size=prev.shape + size)
        self.neurons = np.random.uniform(low=-1, high=1, size=prev.shape + size)
        self.bias = np.random.uniform(low=-1, high=1, size=size)
        #self.neurons = np.zeros(prev.shape + size)
        self.prev = prev
        self.activation = activation

    def forward(self):
        self.prev.forward()
        self.sumOutput = (self.prev.output.reshape(self.prev.shape + (1, 1)) * self.neurons).sum(axis=(0,1)) + self.bias
        if self.activation == 'sigmoid':
            self.output = expit(self.sumOutput)
        elif self.activation == 'linear':
            self.output = self.sumOutput
        else:
            raise ValueError("No such activation function: " + str(self.activation))

    def backward(self, next_deriv: np.ndarray, learn_rate: float):
        if self.activation == 'sigmoid':
            sigmoid_diff = (1 / (np.exp(self.sumOutput) + np.exp(-self.sumOutput)) ** 2)
        elif self.activation == 'linear':
            sigmoid_diff = np.ones(self.sumOutput.shape)
        else:
            raise ValueError("No such activation function: " + str(self.activation))
        self.bias -= learn_rate * sigmoid_diff * next_deriv
        sigmoid_diff = sigmoid_diff.reshape((1,1) + sigmoid_diff.shape)
        prev_outputs = self.prev.output.reshape(self.prev.output.shape + (1,1))
        next_deriv = next_deriv.reshape((1, 1) + next_deriv.shape)
        self.prev.backward((next_deriv * sigmoid_diff * self.neurons).sum(axis=(2,3)), learn_rate)

        minus = sigmoid_diff * prev_outputs * next_deriv
        #print(self.activation, minus.min(), minus.mean(), minus.max())
        self.neurons -= learn_rate * minus

"""
Represents a convolution network layer.
Implementation calls scipy.signal.convolve2d(mode="valid") to perform the convolution.
"""
class ConvolutionLayer(NetworkLayer):
    """
    Initializes a new instance of ConvolutionLayer.
    Parameters:
        * prev - previous layer of the network (a layer that produces results which are further processed by this layer)
        * kernel - size of the convolution kernel
    """
    def __init__(self, prev: NetworkLayer, kernel: tuple):
        super(ConvolutionLayer, self).__init__((prev.shape[0] - kernel[0] + 1, prev.shape[1] - kernel[1] + 1))
        self.neurons = np.random.uniform(low=-1, high=1, size=kernel)
        self.prev = prev

    def forward(self):
        self.prev.forward()
        self.sumOutput = convolve2d(self.prev.output, self.neurons, mode="valid")
        self.output = expit(self.sumOutput)

    def backward(self, next_deriv: np.ndarray, learn_rate: float):
        sigmoid_diff = (1 / (np.exp(self.sumOutput) + np.exp(-self.sumOutput)) ** 2)
        self.prev.backward(convolve2d(next_deriv * sigmoid_diff, self.neurons[::-1, ::-1], mode="valid"), learn_rate)
        self.neurons -= learn_rate * convolve2d(self.prev.output, next_deriv * sigmoid_diff, mode="valid")

"""
Represents a linear subsampling network layer.
"""
class SubsampleLayer(NetworkLayer):
    """
    Initializes a new instance of SubsampleLayer.
    Parameters:
        * prev - previous layer of the network (a layer that produces results which are further processed by this layer)
        * kernel - subsampling rate. Must be divisors of prev.shape
    """
    def __init__(self, prev: NetworkLayer, kernel: tuple):
        assert prev.shape[0] % kernel[0] == 0 and prev.shape[1] % kernel[1] == 0
        super(SubsampleLayer, self).__init__((prev.shape[0] // kernel[0], prev.shape[1] // kernel[1]))
        self.prev = prev
        self.kernel = kernel

    def forward(self):
        self.prev.forward()
        self.output = self.prev.output.reshape((self.prev.shape[0] // self.kernel[0], self.kernel[0], self.prev.shape[1] // self.kernel[1], self.kernel[1])).mean(axis=(1, 3))

    def backward(self, next_deriv: np.ndarray, learn_rate: float):
        next_deriv = np.expand_dims(np.expand_dims(next_deriv, 1), 3)
        next_deriv = np.concatenate([np.concatenate([next_deriv] * self.kernel[1], axis=3)] * self.kernel[0], axis=1).reshape(self.prev.shape)
        self.prev.backward(next_deriv / (self.kernel[0] * self.kernel[1]), learn_rate)

"""
Represents a network layer that combines outputs of several layers into a single output array.
"""
class CombineLayer(NetworkLayer):
    """
    Initializes a new instance of CombineLayer.
    Parameters:
        * args - layers which are combined by this layer. Must have an equal shape[1].
    """
    def __init__(self, *args):
        assert len(set(map(lambda a: a.shape[1], args))) == 1
        super(CombineLayer, self).__init__((sum(map(lambda a: a.shape[0], args)), args[0].shape[1]))
        self.prev = args

    def set_input(self, input):
        self.prev[0].set_input(input)

    def forward(self):
        for a in self.prev:
            a.forward()
        self.output = np.vstack(a.output for a in self.prev)

    def backward(self, next_deriv: np.ndarray, learn_rate: float):
        start = 0
        for a in self.prev:
            a.backward(next_deriv[start:start+a.shape[0], :], learn_rate)
            start += a.shape[0]
