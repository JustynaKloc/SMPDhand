#działa
import csv
import numpy
import math
import random
from sfs import *
from fisher import *

filePath = 'lista.txt'


''' subset - na ile zbiorow dzielimy probki, iterations - ile iteracji'''
def crossvalidate(subsets, iterations, BestFeatures):
    
    for i in range (0, iterations):
        trainSet = []
        testSet = []
        totalSet = []

        with open(filePath, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if "prosta" in row[0] or "piesc" in row[0]:
                    totalSet.append(row)

        selectedBestFeatures = BestFeatures.copy()
        selectedBestFeatures = numpy.array(selectedBestFeatures) + 1
        if 0 not in selectedBestFeatures:
            selectedBestFeatures = numpy.insert(selectedBestFeatures, 0, [0])
        a = numpy.array(totalSet)
        e = a[:, selectedBestFeatures]

        totalSetCopy = e.tolist()
        dividedSet = []

        for i in range(subsets):
            dividedSet.append([]) 
        
        set = 0
        for i in range(len(totalSetCopy)):
            item = random.choice(totalSetCopy)
            dividedSet[set].append(item)
            totalSetCopy.remove(item)
            set += 1
            if(set == (subsets)):
                set = 0
        
        dividedSetCopy = dividedSet.copy()
        for i in range(subsets, 0, -1):
            testSet = dividedSetCopy[i - 1].copy()
            dividedSetCopy.pop(i - 1)
            for item in dividedSetCopy:
                trainSet.extend(item)
    return trainSet, testSet


#cechy = calculateSFS(4)
#print(crossvalidate(4,1,cechy))
