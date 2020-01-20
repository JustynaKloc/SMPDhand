#ideał
import numpy
import math
from dataload import loadData

filePath = 'lista.txt'


def calculateSFS(features_number):
    A, B = loadData(filePath)
    #wczytuje z podziałem na klasy, tworzę macierze z odpowiednim 
    A = numpy.array(A, dtype=float)
    B = numpy.array(B, dtype=float)
    #dla klas tworzę macierz ze średnimi
    AMeans = numpy.mean(A, axis=0)
    BMeans = numpy.mean(B, axis=0)

    maxFisher = -1.0
    bestIndexes = []
    bestIndex = -1
    NumberOfFeatures = features_number

    # dla jednej cechy
    for index in range(0, 64):
        #obliczam średnie
        AMean = AMeans[index]
        BMean = BMeans[index]
        #odejmuję średnie
        AValues = A[:, index] - AMean
        BValues = B[:, index] - BMean
        #obliczam odchylenie standardowe
        odchylenieA = math.sqrt((sum([i ** 2 for i in AValues]))/A.shape[0])
        odchylenieB = math.sqrt((sum([i ** 2 for i in BValues]))/B.shape[0])
        #oblicza współczynnik Fishera według wzoru
        fisher = abs(AMean - BMean)/(odchylenieA +  odchylenieB)
        if fisher > maxFisher:
            maxFisher = fisher
            bestIndex = index
    bestIndexes.append(bestIndex)
    #przypadek wielu cech
    if(NumberOfFeatures > 1):
        while len(bestIndexes) < NumberOfFeatures:
            for index in range(0, 64):

                if index in bestIndexes:
                    continue
                AMean = []
                BMean = []
                AValues = []
                BValues = []

                combinacjacech = bestIndexes.copy()
                combinacjacech.append(index)

                for cecha in combinacjacech:
                    
                    AValues.append(A[:, cecha] - AMeans[cecha])
                    BValues.append(B[:, cecha] - BMeans[cecha])
                    #licze srednia
                    AMean.append(AMeans[cecha])
                    BMean.append(BMeans[cecha])
                #liczę kowariancję, a następnie według wzoru wyliczam cechy
                ACovariance = (1 / A.shape[0]) * numpy.dot(numpy.array(AValues, dtype=float),
                                                                     numpy.transpose(numpy.array(AValues, dtype=float)))
                BCovariance = (1 / B.shape[0]) * numpy.dot(numpy.array(BValues, dtype=float),
                                                                      numpy.transpose(numpy.array(BValues, dtype=float)))

                fisher = numpy.linalg.norm(numpy.array(AMean, dtype=float) - numpy.array(BMean, dtype=float)) / \
                                numpy.linalg.det(ACovariance + BCovariance)
                if fisher > maxFisher:
                    maxFisher = fisher
                    bestIndex = index

            bestIndexes.append(bestIndex)
    return bestIndexes

#print(calculateSFS(4))