import numpy as np
import re
from Methods_Preprocessing import preprocessor1

def sibAveraging(first, second):
    return [ ((x + y)/2.0) for x, y in zip(first, second)]

def WordAveraging(sent, WV, dim):
    summ = [0.0] * dim
    A = 0.0;
    sent_A=preprocessor1(re.sub(r"[\n(\[\])]", "", sent)).split(" ")
    for word in sent_A:
        if word in WV : #and word not in stop:
            A = A + 1.0
            for i in range(0, len(WV[word])):
                summ[i] = summ[i] + float((WV[word])[i])
    if A != 0:
        for i in range(0, dim):
            summ[i] = summ[i] / A
    return summ;

def readWV(path, stop):
    WV = {}
    file_1 = open(path, "r")
    for line in file_1:
        line = line.replace("\n", "")
        wordV = line.split(" ")
        key = wordV[0]

        if key not in stop:
            del wordV[0]
            WV[key] = (np.asarray(wordV,dtype=float))

    return WV