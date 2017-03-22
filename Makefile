CPPFLAGS=-I/usr/include/eigen3 --std=c++11 -g -O2 -msse
LFLAGS=-llapack -fopenmp

all: main unit_test

main: main.o.o hello.o
    g++ main.o $(LFLAGS) -o main

main.o: main.cpp
    g++ -c $(CPPFLAGS) -o main.o main.cpp

unit_test: unit_test.o
    g++ unit_test.o $(LFLAGS) -o unit_test

unit_test.o: unit_test.cpp
    g++ -c $(CPPFLAGS) -o main.o main.cpp

clean:
    rm main.o unit_test.o main unit_test
