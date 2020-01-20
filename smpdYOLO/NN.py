#działa
import numpy
import math
import sys
from bootstrap import *
from sfs import *
from fisher import *

def NN(trainSet, testSet):
    #kopiuje zbiór testowy i treningowy 
    testSetCopy = testSet.copy()
    trainSetCopy = trainSet.copy()
    #zmienne na sumowanie ilości dokonanych dobrych i złych klasyfikacji
    goodClassification = 0
    badClassification = 0
    # dla każdego elementu ze zbioru testowego
    for testItem in testSetCopy:
        minDistance = sys.maxsize
        classifiedToClass = ""
        for trainItem in trainSetCopy:
            # pomijam nazwe klasy
            difference = numpy.array(testItem[1:], dtype=float) - numpy.array(trainItem[1:], dtype=float)
            squared = []
            # liczę odległości euklidesa
            for i in difference:
                squared.append(pow(i, 2))
            distance = math.sqrt(numpy.sum(numpy.array(squared)))
            if distance < minDistance:
                minDistance = distance
                classifiedToClass = trainItem[0]
        #segreguję dobrze przydzielone i źle przydzielone próbki    
        if "prosta" in testItem[0] and "prosta" in classifiedToClass:
            goodClassification = goodClassification + 1
        elif "piesc" in testItem[0] and "piesc" in classifiedToClass:
            goodClassification = goodClassification + 1
        else:
            badClassification = badClassification + 1
    #wyliczam skuteczność algorytmu 
    goodClassificationPercent = 100 * goodClassification / numpy.array(testSetCopy).shape[0]
    return goodClassificationPercent

#cechy = calculateSFS(4)
#train, test = bootstrap(1, 70, cechy)
#A = NN(train, test)
#print(A)