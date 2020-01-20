#działa
import numpy
import math
from sfs import *
from fisher import *
from bootstrap import *
from crossvalidation import *

filePath = 'lista.txt'


def NM(trainSet, testSet):
    testSetCopy = testSet.copy()
    trainSetCopy = trainSet.copy()
    goodClassification = 0
    badClassification = 0
    ATrainSet = []
    BTrainSet = []

    for item in trainSetCopy:
        if "prosta" in item[0]:
            # del item[0]
            ATrainSet.append(item[1:])
        elif "piesc" in item[0]:
            # del item[0]
            BTrainSet.append(item[1:])
    # policz średnią klasy
    AMean = []
    BMean = []
    AMean = numpy.mean(numpy.array(ATrainSet, dtype=float), axis=0)
    BMean = numpy.mean(numpy.array(BTrainSet, dtype=float), axis=0)

    for testItem in testSetCopy:
        # policz odległość od średniej klasy prosta i piesc
        differenceA = numpy.array(testItem[1:], dtype=float) - numpy.array(AMean, dtype=float)
        differenceB = numpy.array(testItem[1:], dtype=float) - numpy.array(BMean, dtype=float)

        squaredA = []
        squaredB = []
        for i in differenceA:
            squaredA.append(pow(i, 2))
        for i in differenceB:
            squaredB.append(pow(i, 2))
        distanceA = math.sqrt(numpy.sum(numpy.array(squaredA)))
        distanceB = math.sqrt(numpy.sum(numpy.array(squaredB)))

        if distanceA < distanceB:
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


cechy = calculateSFS(4)
train, test = bootstrap(1, 70, cechy)
A = NM(train, test)
print(A)