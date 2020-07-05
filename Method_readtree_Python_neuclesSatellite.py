import os
import re
import ClassNode
import numpy as np
import math



from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

def sortEduKey(eduKeys, reverse= False):
    if reverse:
        eduKeys = [int(x) for x in eduKeys]
        eduKeys = (sorted(eduKeys, reverse=True))
        return eduKeys

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    #if np.sum(x) == 0:
        #return np.zeros(x.shape)
    return 1.0 - np.tanh(x)**2

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

def readWV(path):
    WV = {}
    file_1 = open(path, "r")
    for line in file_1:
        line = line.replace("\n", "")
        wordV = line.split(" ")
        key = wordV[0]

        if key not in stop:
            del wordV[0]
            #print wordV
            WV[key] = (np.asarray(wordV,dtype=float))
            #print WV[key]
    return WV


def feedforward_act(ab , W1, activationFunc):
    # if len(ab.shape())==1:
    #     ab=np.expand_dims(ab,0)
    #print ab

    if(activationFunc=="tanh"):
        #print W1.shape
        #print np.matrix(ab).shape
        # if len((np.matmul(np.array(ab),W1)))== 1:
        #     return (np.tanh(np.matmul(np.array(ab),W1)).tolist())[0]
        return np.tanh(np.matmul(np.array(ab),W1))
    # elif (activationFunc == "sig"):
    #     return sigmoid(np.matmul(ab, W1))
    # elif (activationFunc == "relu"):
    #     return ReLU(np.matmul(ab, W1))
    # else:
    #     return (np.matmul(ab, W1))


def preprocessor1(words):
    pre_word=re.sub(r"[^a-zA-Z]", " ", words.lower())
    #print pre_word
    return pre_word

def sibAveraging(first, second):
    return [ ((x + y)/2.0) for x, y in zip(first, second)]

def WordAveraging(sent, WV, dim):
    summ = [0.0] * dim
    A = 0.0;
    sent_A=preprocessor1(re.sub(r"[\n(\[\])]", "", sent)).split(" ")
    for word in sent_A:
        if word in WV and word not in stop:
            A = A + 1.0
            for i in range(0, len(WV[word])):
                summ[i] = summ[i] + float((WV[word])[i])
    if A != 0:
        for i in range(0, dim):
            summ[i] = summ[i] / A
    return summ;


def readTree(filePath, W1, WV, dim, WSat, WNeu):
    #print filePath
    EDUs = {}
    EDUs_main = {}
    if os.path.exists(filePath):
        if  os.stat(filePath).st_size > 0 :
            file_1 = open(filePath, "r")
            for line in file_1:

                line = re.sub(r"[\n(\[\])]", "", line)
                arr = re.split("\s*,\s*",line)
                arr2 = re.split("\'", line)

                relation = arr2[len(arr2)-2]
                #node.nodeRelation = relation
                hierarchy = arr2[len(arr2)-4]
                #node.nodeHierarchy = hierarchy

                if arr[0] == arr[1]:
                    if int(arr[0]) / 10 != 0:
                        arr[0] = "9" + arr[0]
                    vector = WordAveraging(preprocessor1(re.sub(r"[\n,.:'(\[\])]", "", arr2[1])), WV, dim)
                    #if "910" in numconcat2:
                    #print len(vector)
                    #print len(vector)
                    if hierarchy == "Nucleus" :
                        vector = feedforward_act(vector, WNeu, "tanh")
                    elif hierarchy == "Satellite" :
                        vector = feedforward_act(vector, WSat, "tanh")

                    EDUs_main[arr[0]] = ClassNode.Node(True, False, 0, 0, "", hierarchy, relation, vector, "")
                    EDUs[arr[0]] = "hi"
                else:
                    numconcat = ""
                    for num in range(int(arr[0]), int(arr[1])+1):
                        if num/10 !=0 :
                                num="9"+str(num)
                        numconcat += `int(num)`
                    childs={}
                    i=1

                    numconcat2 = numconcat
                    EDUitem =  sortEduKey(EDUs_main.keys(), reverse=True)
                    #print EDUitem
                    for key in EDUitem:
                        key = str(key)
                        if numconcat!='' and key in numconcat:
                            childs[i] = EDUs_main[key].vector

                            #print numconcat
                            if i==1:
                                leftChild = key
                                EDUs_main[key].child="left"
                            elif i==2:
                                rightChild = key
                                EDUs_main[key].child = "right"
                            i = 2
                            numconcat = numconcat.replace(key,"")
                            del EDUs[key]



                        #print len(childs[1])
                        #childs[1]= (childs[1])[0,:]
                        #print len(childs[1])
                    #
                    # if len(childs[1]) == 1:
                    #     childs[1] = childs[1][0]
                    # if len(childs[2]) == 1:
                    #     childs[2] = childs[2][0]
                    # if "910" in numconcat2:

                        #print len(childs[1])

                    EDUs[numconcat2]="hi"
                    #print W1.shape
                    #print "type", type(childs[1])

                    vector = feedforward_act(np.concatenate([childs[2], childs[1]], 0), W1, "tanh")
                    # print "num", numconcat2
                    # print "len", len(vector)
                    if hierarchy == "Nucleus" :
                        vector = feedforward_act(vector, WNeu, "tanh")
                    elif hierarchy == "Satellite" :
                        vector = feedforward_act(vector, WSat, "tanh")
                    EDUs_main[numconcat2]=ClassNode.Node(False, False, leftChild, rightChild, "" , hierarchy, relation, vector, "")


        #node = ClassNode.Node()
        eduKey=EDUs.keys()
        eduKey.sort()
        if len(eduKey)>1:
            vector = feedforward_act(np.concatenate([EDUs_main[eduKey[0]].vector, EDUs_main[eduKey[0]].vector], 0), W1, "tanh")

            if hierarchy == "Nucleus":
                vector = feedforward_act(vector, WNeu, "tanh")
            elif hierarchy == "Satellite":
                vector = feedforward_act(vector, WSat, "tanh")

            EDUs[eduKey[0]+eduKey[1]] = "hi"
            EDUs_main[eduKey[0]+eduKey[1]] = ClassNode.Node(False, True, eduKey[0], eduKey[1],"", "", "", vector, "")
            del EDUs[eduKey[0]]
            del EDUs[eduKey[1]]






        return EDUs_main

    else:
        return EDUs_main

'''
W1=initialize_weight_variable([200, 100]).value()
WV=readWV("/home/erfaneh/Shared/WordVector/glove.6B/glove.6B.100d.txt")
EDUs = readTree("/home/erfaneh/RST_Parser/Paraphrase/bracketsTxt_Test/629292.bracketsTxt", W1, WV, 100)
print "Yeyyy"
'''