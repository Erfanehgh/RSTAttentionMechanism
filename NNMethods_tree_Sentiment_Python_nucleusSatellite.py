import numpy as np
from Method_readtree_Python import readWV
from Method_readtree_Python import readTree
import os
import math


# def element_wise_sig(c):

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    # if np.sum(x) == 0:
    # return np.zeros(x.shape)
    return np.subtract(1.0, np.tanh(np.array(x)) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigma(x):
    return np.divide((1.0), np.add((1.0), np.exp(np.negative(x))))


def sigmaprime(x):
    return np.multiply(sigmoid(x), np.subtract((1.0), sigmoid(x)))


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def feedforward_act(ab, W1, activationFunc):
    # if len(ab.shape())==1:
    #     ab=np.expand_dims(ab,0)
    if (activationFunc == "tanh"):
        return np.tanh(np.matmul(ab, W1))
    elif (activationFunc == "sig"):
        return tanh(np.matmul(ab, W1))
    elif (activationFunc == "relu"):
        return ReLU(np.matmul(ab, W1))
    else:
        return (np.matmul(ab, W1))


def feedforward(ab, W1):
    # if len(ab.shape)==1:
    # ab=np.expand_dims(ab,0)
    return np.matmul(ab, W1)


def initialize_weight_variable(d1, d2):
    initial = np.divide(np.random.rand(d1, d2).astype(np.float32), 100)
    return initial


def softmax_error(y, y_hat, y_in):
    return np.multiply(np.subtract(y, y_hat), tanh_deriv(y_in))


def non_softmax_error(delta, W2, input, W1):
    # print delta.shape, W2.shape, input.shape, W1.shape
    delta_in = np.matmul(np.matrix(delta), W2.T)
    # print delta_in.shape
    # print tanh_deriv(feedforward(input, W1)).shape
    return np.multiply(delta_in, tanh_deriv(feedforward(input, W1)))


def non_softmax_error_inside(delta, W2, input, W1):
    # print delta.shape, W2.shape, input.shape, W1.shape
    delta_in = np.matmul(np.matrix(delta), W2.T)
    # print delta_in.shape
    # print tanh_deriv(feedforward(input, W1)).shape
    return np.matmul(delta_in, tanh_deriv(feedforward(input, W1)))


def calculate_deltaW(eta, error, input):
    # input3= np.transpose(input2)
    delta_W = np.matmul(np.matrix(input).T, np.matrix(error))
    return np.multiply(eta, delta_W)


def update_weight(W, delta_W):
    # print W.shape
    # print delta_W
    return np.add(W, delta_W)

def attention(EDU, WNeu, WSat):
    vector = EDU.vector
    hierarchy = EDU.hierarchy
    if hierarchy == "Nucleus":
        return feedforward_act(vector, WNeu, "tanh")
    elif hierarchy == "Satellite":
        return feedforward_act(vector, WSat, "tanh")


def BpthroughTree(eta, EDUs, error_soft, W1, W2, WNeu, WSat):

    delta_W_All_N = np.zeros([100, 100])
    delta_W_All_S = np.zeros([100, 100])
    delta_W_All = np.zeros([200, 100])

    eduKeys = sortEduKey(EDUs.keys(), reverse=True)
    parent_node = EDUs[str(eduKeys[0])]





    input = np.concatenate([attention(EDUs[parent_node.leftChild]), attention(EDUs[parent_node.rightChild])], 0)

    # print "input", sess.run(input)
    delta = non_softmax_error(error_soft, W2, input, W1)
    EDUs[parent_node.leftChild].delta = delta
    EDUs[parent_node.rightChild].delta = delta

    deltaw = calculate_deltaW(eta, delta, input)
    # print "deltaw", sess.run(deltaw)
    W22 = np.zeros([100, 100])
    if EDUs[str(key)].child == "left":
        # print "left"
        W22 = W1[0:100, :]
    elif EDUs[str(key)].child == "right":
        # print "right"
        w22 = W1[100:200, :]

    hi_right = EDU[EDUs[str(key)].right].hierarchy
    hi_left = EDU[EDUs[str(key)].right].hierarchy

    if EDU[EDUs[str(key)].right].hierarchy == "Nucleus"
        delta_N = non_softmax_error(delta, , input, WNeu)
    delta_W_All = np.add(deltaw, delta_W_All)
    # print "delta acal", delta.shape

    for key in eduKeys:
        if EDUs[str(key)].isLeaf == False and EDUs[str(key)].isRoot == False:
            input = np.concatenate([EDUs[EDUs[str(key)].leftChild].vector, EDUs[EDUs[str(key)].rightChild].vector], 0)
            # print "input", sess.run(input)
            W22 = np.zeros([100, 100])
            if EDUs[str(key)].child == "left":
                # print "left"
                W22 = W1[0:100, :]
            elif EDUs[str(key)].child == "right":
                # print "right"
                w22 = W1[100:200, :]
            delta = non_softmax_error(EDUs[str(key)].delta, W22, input, W1)
            EDUs[EDUs[str(key)].leftChild].delta = delta
            EDUs[EDUs[str(key)].rightChild].delta = delta
            deltaw = calculate_deltaW(eta, delta, input)
            # print "deltaw", sess.run(deltaw)
            delta_W_All = np.add(deltaw, delta_W_All)
            # print "delta_W_All", sess.run(delta_W_All)

            # if str(key) == parent_node.leftChild:
            #     delta = non_softmax_error(delta, (W1 [0:100, :]))
            # elif str(key) == parent_node.rightChild:
            #     delta = non_softmax_error(delta, (W1 [100:200, :]))
            # print "delta", sess.run(delta)
            # parent_node = EDUs[str(key)]

    return delta_W_All


def sortEduKey(eduKeys, reverse=False):
    if reverse:
        eduKeys = [int(x) for x in eduKeys]
        eduKeys = (sorted(eduKeys, reverse=True))
        return eduKeys


# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=answer, logits=output))
# updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


WV = readWV("/home/erfaneh/Shared/WordVector/glove.6B/glove.6B.100d.txt")
path_Folder = "/home/erfaneh/PycharmProjects/Recursive-neural-networks-TensorFlow/Data/"
posFileList = os.listdir(path_Folder + "pos/")
negFileList = os.listdir(path_Folder + "neg/")

numberofSamples = min(len(posFileList), len(negFileList))
# numberofSamples = 10

W1 = initialize_weight_variable(200, 100)
W2 = initialize_weight_variable(100, 2)
WNeu = initialize_weight_variable(100, 100)
WSat = initialize_weight_variable(100, 100)

delta_W1 = np.zeros([200, 100]).astype(np.float32)
delta_W2 = np.zeros([100, 2]).astype(np.float32)
delta_WNeu = np.zeros([100, 100]).astype(np.float32)
delta_WSat = np.zeros([100, 100]).astype(np.float32)

input2 = np.zeros([1, 100])
y_in = np.zeros([1, 2])
error_soft = np.zeros([1, 2])
error_non_soft = np.zeros([1, 100])
output = np.zeros([1, 2])

OutputFile = open(path_Folder + "output.txt", "w")
eta = 0.005
EDU = {}

for i in range(0, 1000):
    for j in range(0, numberofSamples):
        path_File = path_Folder + "pos/" + posFileList[j]
        EDUs = readTree(path_File, W1, WV, 100)
        y = [1.0, 0]
        eduKeys = sortEduKey(EDUs.keys(), reverse=True)
        input2 = EDUs[str(eduKeys[0])].vector
        y_in = feedforward(input2, W2)
        output = tanh(y_in)
        # print len(y)
        # print output.shape
        # print y_in.shape
        error_soft = softmax_error(y, output, y_in)
        OutputFile.write("%s,%s,%s,%s\n" % (str(i), str(j), "pos", ([output])))
        print  str(i), " ", str(j), " pos ", ([output])

        delta_W2 = calculate_deltaW(eta, error_soft, input2)
        delta_W1 = BpthroughTree(eta, EDUs, error_soft, W1, W2)

        # print delta_W1

        W2 = update_weight(W2, delta_W2)
        W1 = update_weight(W1, delta_W1)
        WNeu = update_weight(WNeu, delta_WNeu)
        WSat = update_weight(WSat, delta_WSat)


        # print W2



        path_File = path_Folder + "neg/" + negFileList[j]
        EDUs = readTree(path_File, W1, WV, 100)
        y = [0, 1.0]
        eduKeys = sortEduKey(EDUs.keys(), reverse=True)
        input2 = EDUs[str(eduKeys[0])].vector
        y_in = feedforward(input2, W2)
        output = tanh(y_in)
        error_soft = softmax_error(y, output, y_in)
        OutputFile.write("%s,%s,%s,%s\n" % (str(i), str(j), "neg", ([output])))
        print  str(i), " ", str(j), " neg ", ([output])

        delta_W2 = calculate_deltaW(error_soft, y_in, input2)
        delta_W1, delta_WNeu, delta_WSat = BpthroughTree(eta, EDUs, error_soft, W1, W2)

        W2 = update_weight(W2, delta_W2)
        W1 = update_weight(W1, delta_W1)
        WNeu = update_weight(WNeu, delta_WNeu)
        WSat = update_weight(WSat, delta_WSat)

OutputFile.close()
