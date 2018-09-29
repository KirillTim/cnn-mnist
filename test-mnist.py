from sklearn.datasets import fetch_mldata
from layer import *
from random import shuffle

mnist = fetch_mldata('MNIST original')

baseLayer = InitialLayer((28, 28))
full1 = FullLayer(baseLayer, (10, 10), 1e-2)
full2 = FullLayer(full1, (5, 5), 1e-2)
result = FullLayer(full2, (1, 10), 1e-2)

a = list(zip(mnist.data, mnist.target))
shuffle(a)
for i, (data, target) in enumerate(a[:7000]):
    if i > 0 and i % 1000 == 0:
        print(i, result.output)
    result.set_input(data.reshape((28, 28)) / 256)
    result.forward()
    lossDeriv = np.where(np.arange(10) == int(target), 1 / (1 - result.output), -1 / result.output)
    result.backward(lossDeriv)

print(full1.neurons)
print(full2.neurons)
print(result.neurons)
