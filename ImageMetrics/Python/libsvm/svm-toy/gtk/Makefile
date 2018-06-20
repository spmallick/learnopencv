CC? = gcc
CXX? = g++
CFLAGS = -Wall -O3 -g `pkg-config --cflags gtk+-2.0`
LIBS = `pkg-config --libs gtk+-2.0`

svm-toy: main.o interface.o callbacks.o ../../svm.o
	$(CXX) $(CFLAGS) main.o interface.o callbacks.o ../../svm.o -o svm-toy $(LIBS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

interface.o: interface.c interface.h
	$(CC) $(CFLAGS) -c interface.c

callbacks.o: callbacks.cpp callbacks.h
	$(CXX) $(CFLAGS) -c callbacks.cpp

../../svm.o: ../../svm.cpp ../../svm.h
	make -C ../.. svm.o

clean:
	rm -f *~ callbacks.o svm-toy main.o interface.o callbacks.o ../../svm.o
