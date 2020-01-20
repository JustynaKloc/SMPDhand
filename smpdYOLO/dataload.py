import csv

filePath = 'lista.txt'

#fukcja wczytująca dane i dzieląca je na dwie klasy (bo tyle jest w pliku)
def loadData(filePath):
    AClass = []
    BClass = []
    with open(filePath, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if "prosta" in row[0]:
                del row[0]
                AClass.append(row)
            if "piesc" in row[0]:
                del row[0]
                BClass.append(row)
    return AClass, BClass

#print(loadData(filePath))