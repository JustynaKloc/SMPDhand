import numpy as np
import random
from bootstrap import *
from sfs import *
import sys

#dystans euklidesa do kwadratu
def distance(p1, p2): 
    return np.sum((p1 - p2)**2) 

def kmeans(data, k): 
    
    #losownie punktu 
    centroids = [] 
    centroids.append(data[np.random.randint(data.shape[0]), :]) 
   
    for c_id in range(k - 1): 
        dist = [] 
        #liczenie dystansu
        for i in range(data.shape[0]): 
            point = data[i, :] 
            d = sys.maxsize 
              
            for j in range(len(centroids)): 
                temp_dist = distance(point, centroids[j]) 
                d = min(d, temp_dist) 
            dist.append(d) 
              
        dist = np.array(dist) 
        next_centroid = data[np.argmax(dist), :] 
        centroids.append(next_centroid) 
        dist = [] 
    return centroids 


def kNM(k, trainSet, testSet):

    testSetCopy = testSet.copy()
    trainSetCopy = trainSet.copy()
    goodClassification = 0
    badClassification = 0
    ATrainSet = []
    BTrainSet = []  
    ATestSet = []
    BTestSet = []  

    #podziel dane według pierwszej wartości 
    for item in trainSetCopy:
        if "prosta" in item[0]:
            ATrainSet.append(item[1:])
        elif "piesc" in item[0]:
            BTrainSet.append(item[1:])

    for item in testSetCopy:
        if "prosta" in item[0]:
            ATestSet.append(item[1:])
        elif "piesc" in item[0]:
            BTestSet.append(item[1:])

    #zamień typ 
    ATrainSet = numpy.array(ATrainSet, dtype=float)
    BTrainSet = numpy.array(BTrainSet, dtype=float)
    
    CenterA = []
    CenterB = []
    #wyznacz centroidy - ilość uzależniona od parrametru k 
    CenterA = kmeans(ATrainSet,k)
    CenterB = kmeans(BTrainSet,k)


    for testItem in testSetCopy:
        # policz odległość od średniej klasy Acer i Quercus
        distances_A = []
        distances_B = []
        squaredA = []
        squaredB = []
        for i in range(0,k):
            differenceA = numpy.array(testItem[1:], dtype=float) - (np.ones(np.size(CenterA[i]))* np.mean(CenterA[i],axis =0))
            differenceB = numpy.array(testItem[1:], dtype=float) - (np.ones(np.size(CenterB[i]))* np.mean(CenterB[i],axis =0))
            for i in differenceA:
                squaredA.append(pow(i, 2))
            for i in differenceB:
                squaredB.append(pow(i, 2))
            distanceA = math.sqrt(numpy.sum(numpy.array(squaredA)))
            distanceB = math.sqrt(numpy.sum(numpy.array(squaredB)))
            #kwalifikacja na podstawie dystansu euklidesa
            if distanceA < distanceB:
                classifiedToClass = "prosta"
            else:
                classifiedToClass = "piesc"
        #zliczenie ilosci nazw 
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
A = kNM(2,train, test)
print(A)