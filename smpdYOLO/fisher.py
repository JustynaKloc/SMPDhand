#działa
import numpy
import math
import itertools
from dataload import *

filePath = 'lista.txt'

def calculateFisher(features_number):
    #wczytuje dane z podziałem na klasy
    A, B = loadData(filePath)
    #zmieniam format danych
    AClass = numpy.array(A, dtype=float)
    BClass = numpy.array(B, dtype=float)
    #obliczam średnie
    AMeans = numpy.mean(AClass, axis=0)
    BMeans = numpy.mean(BClass, axis=0)

    maxFisher =  0.0
    bestIndexes = []
    best = []
    
    #ilość cech które mam znaleźć
    NumberOfFeatures = features_number
    #w przypadku gdy cecha jest równa 1
    if(NumberOfFeatures == 1):
        for index in range(0, 64):
            
            AMean = AMeans[index]
            BMean = BMeans[index]

            AValues = AClass[:, index] - AMean
            BValues = BClass[:, index] - BMean

            #liczę ochylenie standardowe dla klas 
            AOdchylenie = math.sqrt((sum([i ** 2 for i in AValues]))/AClass.shape[0])
            BOdchylenie = math.sqrt((sum([i ** 2 for i in BValues]))/BClass.shape[0])
            #podstawiam do wzoru 
            fisher = abs(AMean - BMean)/(AOdchylenie + BOdchylenie)
            if fisher > maxFisher:
                maxFisher = fisher
                bestIndex = index

        return bestIndex
    #przypadek gdy cech jest więcej niż jedna
    else:
        #kombinacje z cech 
        for combination in itertools.combinations([i for i in range(0, 64)], selectedNumberOfFeatures):
            AMean = []
            BMean = []
            AValues = []
            BValues = []

            for feature in combination:
                AValues.append(AClass[:, feature] - AMeans[feature])
                BValues.append(BClass[:, feature] - BMeans[feature])
                AMean.append(AMeans[feature])
                BMean.append(BMeans[feature])


            ACovariance = (1 / AClass.shape[0]) * numpy.dot(numpy.array(AValues, dtype=float),
                                                                 numpy.transpose(numpy.array(AValues, dtype=float)))
            BCovariance = (1 / BClass.shape[0]) * numpy.dot(numpy.array(BValues, dtype=float),
                                                                  numpy.transpose(numpy.array(BValues, dtype=float)))

            fisher = numpy.linalg.norm(numpy.array(AMean, dtype=float) - numpy.array(BMean, dtype=float)) / \
                            numpy.linalg.det(ACovariance + BCovariance)

            if fisher > maxFisher:
                maxFisher = fisher
                bestIndexes = combination
        #tylko po to aby zwrócić listę 
        for c in bestIndexes:
            best.append(c)

        return  best

#print(calculateFisher(2))