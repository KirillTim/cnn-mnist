from sklearn.datasets import fetch_mldata
from layer import *
from random import shuffle

mnist = fetch_mldata('MNIST original')

baseLayer = InitialLayer((28, 28))
full1 = FullLayer(baseLayer, (10, 10), 1e-2)
result = FullLayer(full1, (1, 10), 1e-2, activation='linear')

a = list(zip(mnist.data, mnist.target))
shuffle(a)
lossFunc = None
for i, (data, target) in enumerate(a[:60000]):
    if i > 0 and i % 1000 == 0:
        print(i, lossFunc)
    result.set_input(data.reshape((28, 28)) / 256)
    result.forward()
    exp = np.exp(result.output)
    sumExp = exp.sum()
    probs = exp / sumExp
    sumExpMinus = sumExp - exp
    label = np.where(np.arange(0, 10) == int(target), 1, 0)
    sumSecond = (1 - label) / sumExpMinus
    sumSecond = sumSecond.sum() - sumSecond
    cLoss = 10 * np.log(sumExp) - (label * result.output + (1 - label) * np.log(sumExpMinus)).sum()
    lossFunc = cLoss if lossFunc is None else lossFunc * 0.999 + cLoss * 0.001
    lossDeriv = 10 * probs - label - exp * sumSecond
    result.backward(lossDeriv)

lossFunc = 0
nHits = 0
for data, target in a[60000:]:
    result.set_input(data.reshape((28, 28)) / 256)
    result.forward()
    if np.argmax(result.output) == int(target):
        nHits += 1
    exp = np.exp(result.output)
    sumExp = exp.sum()
    sumExpMinus = sumExp - exp
    label = np.where(np.arange(0, 10) == int(target), 1, 0)
    lossFunc += 10 * np.log(sumExp) - (label * result.output + (1 - label) * np.log(sumExpMinus)).sum()
print('Test loss', lossFunc / (len(a) - 60000), 'accuracy', nHits / (len(a) - 60000))

#print(full1.neurons)
#print(result.neurons)
