from PyQt4.QtGui import *
from PyQt4 import uic
from fisher import *
from sfs import *
from bootstrap import *
from crossvalidation import * 
from NN import *
from kNN import *
from NM import *
from kNM import *


#zmienna przechowuje cechy które zostały wybrane za pomocą fishera lub sfs
wybranecechy = []

#funkcja która w zależności od wybranych algorytmów wywołuje odpowiednie funkcje
def getciag():
    ilcech = dlg.liczbacech.text()
    i = int(ilcech)
    if dlg.fisher.isChecked():
        nowy = calculateFisher(i)
        print(nowy)
    else:
        nowy = calculateSFS(i)
        print(nowy)
    wybranecechy = nowy
    if dlg.cv.isChecked():
        czesci = dlg.cvczesci.text()
        czesci = int(czesci)
        ite = dlg.cviter.text()
        ite = int(ite)
        train, test = crossvalidate(czesci,ite,wybranecechy)
    else:
        procent = dlg.bsprocent.text()
        procent = int(procent)
        ite = dlg.bsiter.text()
        ite = int(ite)
        train, test = bootstrap(ite,procent,wybranecechy)
    if dlg.NN.isChecked():
        wynik = NN(train,test)
        wynik = str(wynik)
    elif dlg.NM.isChecked():
        wynik = NM(train,test)
        wynik = str(wynik)
    elif dlg.kNN.isChecked():
        ks = dlg.liczbasas.text()
        ks = int(ks)
        wynik = kNN(train,test,ks)
        wynik = str(wynik)
    elif dlg.kNM.isChecked():
        ks = dlg.liczbasas.text()
        ks = int(ks)
        wynik = kNM(ks, train,test)
        wynik = str(wynik)
    dlg.skutek.setText(wynik)
 

app = QApplication([])
#plik stworzony w QT4 Designer 
dlg = uic.loadUi("smpd.ui")
#wywołanie funkcji po zaakceptowaniu 
dlg.ok.clicked.connect(getciag)

dlg.show()
app.exec()