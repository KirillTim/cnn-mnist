import numpy as np
from scipy.special import expit

class NetworkLayer(object):
    pass

class InitialLayer(NetworkLayer):
    def set_input(self, input):
        self.output = input

class FullLayer(NetworkLayer):
    def __init__(self, prev: NetworkLayer, size: tuple, learn_rate: float):
        self.neurons = np.zeros(prev.shape + size)
        self.prev = prev
        self.learn_rate = learn_rate

    def forward(self):
        self.sumOutput = (self.prev.output * self.neurons).sum(axis=(0,1))
        self.output = expit(self.sumOutput)

    def backward(self, next_deriv: np.ndarray) -> np.ndarray:
        sigmoidDiff = (1 / (np.exp(self.sumOutput) + np.exp(-self.sumOutput)) ** 2)
        sigmoidDiff = sigmoidDiff.reshape((1,1) + sigmoidDiff.shape)
        prevOutputs = self.prev.output.reshape(self.prev.output.shape + (1,1))
        next_deriv = next_deriv.reshape((1, 1) + next_deriv.shape)
        result = (next_deriv * sigmoidDiff * self.neurons).sum(axis=(2,3))
        self.neurons -= self.learn_rate * sigmoidDiff * prevOutputs * next_deriv
        return result

