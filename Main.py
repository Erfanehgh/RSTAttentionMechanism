import numpy as np
from Method_WV import readWV
from Method_NeuralNets import initialize_weight_variable
from Methods_Classification import train_AttWeight, test_AttWeight
import os
import math

'''

'''
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

WV=readWV("/path of glove folder/glove.6B.100d.txt", stop)
path="path of Parsed imdb folder"
path_Folder=path+"test/"
path_Folder_test = path+"train/"

eta = 0.001
dim = 100
OutputFile = open("output-attweight-sub-500.csv", "w")


# attScalar

OutputFile.write("attScalar, iteration, data, tp, tn, fp, fn, accuracy, precision, recall, F-1 measure\n")
attOn = "Nucleus"
for attScalar in np.arange(1.0, 2.0, 0.1):

    W1 = initialize_weight_variable(200, 100)
    W2 = initialize_weight_variable(100, 2)

    for i in range (0,40):

        W1, W2 = train(path_Folder, WV, dim, W1, W2, eta, OutputFile, attOn, attScalar)

        if i%1 == 0:
            test(path_Folder, "train", WV, dim, W1, W2, OutputFile, attOn, attScalar, i)
            test(path_Folder_test, "test", WV, dim, W1, W2, OutputFile, attOn, attScalar, i)

'''
Attention Weight
'''
OutputFile.write("iteration, data, tp, tn, fp, fn, accuracy, precision, recall, F-1 measure\n")
activationFuc = "tanh"
W1 = initialize_weight_variable(200, 100)
W2 = initialize_weight_variable(100, 2)
WSat = initialize_weight_variable(100, 100)
WNu = initialize_weight_variable(100, 100)

for i in range(0, 500):

    W1, W2, WSat, WNu = train_AttWeight(path_Folder, WV, dim, W1, W2, eta, OutputFile, WSat, WNu, activationFuc)
    #print WNu [0,:]
    if i % 1 == 0:

        test_AttWeight(path_Folder, "train", WV, dim, W1, W2, OutputFile, WSat, WNu, i, activationFuc)
        test_AttWeight(path_Folder_test, "test", WV, dim, W1, W2, OutputFile, WSat, WNu, i, activationFuc)



OutputFile.close()
