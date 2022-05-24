CXX? = g++
INCLUDE = /usr/include/qt4
CFLAGS = -Wall -O3 -I$(INCLUDE) -I$(INCLUDE)/QtGui -I$(INCLUDE)/QtCore
LIB = -lQtGui -lQtCore
MOC = /usr/bin/moc-qt4

svm-toy: svm-toy.cpp svm-toy.moc ../../svm.o
	$(CXX) $(CFLAGS) svm-toy.cpp ../../svm.o -o svm-toy $(LIB)

svm-toy.moc: svm-toy.cpp
	$(MOC) svm-toy.cpp -o svm-toy.moc

../../svm.o: ../../svm.cpp ../../svm.h
	make -C ../.. svm.o

clean:
	rm -f *~ svm-toy svm-toy.moc ../../svm.o

