from sklearn.datasets import fetch_mldata
from layer import *
from random import shuffle

mnist = fetch_mldata('MNIST original')

baseLayer = InitialLayer((28, 28))
conv1 = ConvolutionLayer(baseLayer, (7, 7))
conv2 = ConvolutionLayer(baseLayer, (7, 7))
conv3 = ConvolutionLayer(baseLayer, (7, 7))
conv4 = ConvolutionLayer(baseLayer, (7, 7))
sub1 = SubsampleLayer(conv1, (2, 2))
sub2 = SubsampleLayer(conv2, (2, 2))
sub3 = SubsampleLayer(conv3, (2, 2))
sub4 = SubsampleLayer(conv4, (2, 2))
comb = CombineLayer(sub1, sub2, sub3, sub4)
full1 = FullLayer(comb, (10, 10))
result = FullLayer(full1, (1, 10), activation='linear')

a = list(zip(mnist.data, mnist.target))
shuffle(a)
lossFunc = None
prevLossFunc = None
learnRate = 0.1
for i, (data, target) in enumerate(a[:60000]):
    if i > 0 and i % 1000 == 0:
        print(i, lossFunc, learnRate)
        if prevLossFunc is not None and lossFunc > prevLossFunc:
            learnRate *= 0.5
        prevLossFunc = lossFunc
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
    result.backward(lossDeriv, learnRate)

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
