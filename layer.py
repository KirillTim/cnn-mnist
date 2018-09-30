import numpy as np
from scipy.special import expit
from scipy.signal import convolve2d

class NetworkLayer(object):
    def __init__(self, shape: tuple):
        self.shape = shape

    def set_input(self, input):
        self.prev.set_input(input)

    def forward(self):
        pass

    def backward(self, next_deriv: np.ndarray):
        pass

class InitialLayer(NetworkLayer):
    def __init__(self, shape):
        super(InitialLayer, self).__init__(shape)
    
    def set_input(self, input):
        self.output = input

class FullLayer(NetworkLayer):
    def __init__(self, prev: NetworkLayer, size: tuple, learn_rate: float, activation='sigmoid'):
        super(FullLayer, self).__init__(size)
        #self.neurons = np.random.normal(size=prev.shape + size)
        self.neurons = np.random.uniform(low=-1, high=1, size=prev.shape + size)
        #self.neurons = np.zeros(prev.shape + size)
        self.prev = prev
        self.learn_rate = learn_rate
        self.activation = activation

    def forward(self):
        self.prev.forward()
        self.sumOutput = (self.prev.output.reshape(self.prev.shape + (1, 1)) * self.neurons).sum(axis=(0,1))
        if self.activation == 'sigmoid':
            self.output = expit(self.sumOutput)
        elif self.activation == 'linear':
            self.output = self.sumOutput
        else:
            raise ValueError("No such activation function: " + str(self.activation))

    def backward(self, next_deriv: np.ndarray):
        if self.activation == 'sigmoid':
            sigmoid_diff = (1 / (np.exp(self.sumOutput) + np.exp(-self.sumOutput)) ** 2)
            sigmoid_diff = sigmoid_diff.reshape((1,1) + sigmoid_diff.shape)
        elif self.activation == 'linear':
            sigmoid_diff = np.ones(self.sumOutput.shape)
        else:
            raise ValueError("No such activation function: " + str(self.activation))
        prev_outputs = self.prev.output.reshape(self.prev.output.shape + (1,1))
        next_deriv = next_deriv.reshape((1, 1) + next_deriv.shape)
        self.prev.backward((next_deriv * sigmoid_diff * self.neurons).sum(axis=(2,3)))

        minus = sigmoid_diff * prev_outputs * next_deriv
        #print(self.activation, minus.min(), minus.mean(), minus.max())
        self.neurons -= self.learn_rate * minus

class ConvolutionLayer(NetworkLayer):
    def __init__(self, prev: NetworkLayer, kernel: tuple, learn_rate: float):
        super(ConvolutionLayer, self).__init__((prev.shape[0] - kernel[0] + 1, prev.shape[1] - kernel[1] + 1))
        self.neurons = np.zeros(kernel)
        self.prev = prev
        self.learn_rate = learn_rate

    def forward(self):
        self.prev.forward()
        self.sumOutput = convolve2d(self.prev.output, self.neurons, mode="valid")
        self.output = expit(self.sumOutput)

    def backward(self, next_deriv: np.ndarray):
        sigmoid_diff = (1 / (np.exp(self.sumOutput) + np.exp(-self.sumOutput)) ** 2)
        self.prev.backward(convolve2d(next_deriv * sigmoid_diff, self.neurons[::-1, ::-1], mode="valid"))
        self.neurons -= self.learn_rate * convolve2d(self.prev.output, next_deriv * sigmoid_diff, mode="valid")
