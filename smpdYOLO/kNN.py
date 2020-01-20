#działa
import numpy
import math
import sys
import heapq
from bootstrap import *
from sfs import *

filePath = 'lista.txt'

'''dane treningowe, dane testowe, ilość sąsiadów '''
def kNN(trainSet, testSet, k):

    testSetCopy = testSet.copy()
    trainSetCopy = trainSet.copy()
    goodClassification = 0
    badClassification = 0


    for testItem in testSetCopy:
        distanceList = []

        for trainItem in trainSetCopy:
            difference = numpy.array(testItem[1:], dtype=float) - numpy.array(trainItem[1:], dtype=float)
            squared = []
            for i in difference:
                squared.append(pow(i, 2))
            distance = math.sqrt(numpy.sum(numpy.array(squared)))
            # odległość próbki od elementu klasy (zapisuje odległość i klase próbki treningowej)
            distanceClassItem = [trainItem[0], distance]
            distanceList.append(distanceClassItem)
        bestDistancesList = heapq.nsmallest(k, distanceList, key=lambda item: item[1])
        AOccurrence = 0
        BOccurrence = 0

        for item in bestDistancesList:
            if "prosta" in item[0]:
                AOccurrence = AOccurrence + 1
            elif "piesc" in item[0]:
                BOccurrence = BOccurrence + 1

        if AOccurrence > BOccurrence:
            classifiedToClass = "prosta"
        else:
            classifiedToClass = "piesc"

        if "prosta" in testItem[0] and "prosta" in classifiedToClass:
            goodClassification = goodClassification + 1
        elif "piesc" in testItem[0] and "piesc" in classifiedToClass:
            goodClassification = goodClassification + 1
        else:
            badClassification = badClassification + 1

    goodClassificationPercent = 100 * goodClassification / numpy.array(testSetCopy).shape[0]
    return goodClassificationPercent

#cechy = calculateSFS(4)
#train, test = bootstrap(1, 70, cechy)
#A = kNN(train, test, 2)
#print(A)