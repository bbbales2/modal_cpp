CPPFLAGS=-fopenmp -I/usr/include/eigen3 --std=c++11 -g -O2 -msse
LFLAGS=-llapack -fopenmp
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
	g++ $< $(LFLAGS) -o $@

unit_tests1: unit_tests1.o
	g++ $< $(LFLAGS) -o $@

unit_tests2: unit_tests2.o
	g++ $< $(LFLAGS) -o $@

clean:
	-rm main.o unit_tests1.o unit_tests2.o main unit_tests1 unit_tests2
