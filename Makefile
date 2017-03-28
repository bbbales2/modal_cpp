CPPFLAGS=-fopenmp -I/usr/include/eigen3 -I/home/bbales2/local/cuda/include -I/home/bbales2/magma-2.2.0/include --std=c++11 -g -msse
LFLAGS=-Xcompiler "-fopenmp " -L/home/bbales2/magma-2.2.0/lib -lmagma -llapack -lblas -lcublas -lcusparse
DEPS=util.hpp mechanics.hpp polybasis.hpp
SOURCES=main.cpp unit_tests1.cpp unit_tests2.cpp

.PHONY: clean all

all: main unit_tests1 unit_tests2

%.o: %.c
	g++ $(CPPFLAGS) -c -o $@ $<

depend: .depend

.depend: $(SOURCES)
	rm -f ./.depend
	g++ $(CPPFLAGS) -MM $^>>./.depend;

include .depend

main: main.o
	nvcc $< -o $@ $(LFLAGS)

unit_tests1: unit_tests1.o
	nvcc $< -o $@ $(LFLAGS)

unit_tests2: unit_tests2.o
	nvcc $< -o $@ $(LFLAGS)

clean:
	-rm main.o unit_tests1.o unit_tests2.o main unit_tests1 unit_tests2
