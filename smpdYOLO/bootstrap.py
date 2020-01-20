#działa
import csv
import numpy
import random
import math
from sfs import calculateSFS
from fisher import *

filePath = 'lista.txt'

''' iteration - ile iteracji train_set_percent - procent próbek na zbiór treningowy, BestFeatures - cechy wybrane przez fishera  '''
def bootstrap(iterations, train_set_percent, BestFeatures):
    nnQuality = []
    knnQuality = []
    nmQuality = []

    #pobierz dane i zamieszaj
    for i in range (0, iterations):
        trainSet = []
        testSet = []
        totalSet = []
        # pobierz dane do trenowania i testowania
        with open(filePath, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if "prosta" in row[0] or "piesc" in row[0]:
                    totalSet.append(row)

        numOfTrainSamples = math.ceil(train_set_percent/100*numpy.array(totalSet).shape[0])

        selectedBestFeatures = BestFeatures.copy()
        selectedBestFeatures = numpy.array(selectedBestFeatures) + 1
        if 0 not in selectedBestFeatures:
            selectedBestFeatures = numpy.insert(selectedBestFeatures, 0, [0])
        a = numpy.array(totalSet)
        e = a[:, selectedBestFeatures]

        testSet = e.tolist()
        # - losujemy próbki z całego zbioru ze zwracaniem(po prostu ich nie usuwamy) i wrzucamy do train seta
        for i in range(0, numOfTrainSamples):
            # wylosuj randomowo item do testu
            item = random.choice(testSet)
            # dodaj do train seta
            trainSet.append(item)
        return trainSet, testSet


#cechy = calculateSFS(4)
#print( bootstrap(1,70,cechy))
