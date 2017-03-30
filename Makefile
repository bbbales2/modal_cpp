CPPFLAGS=-fopenmp -I/home/bbales2/stan_math/lib/boost_1.62.0 -I/home/bbales2/stan_math/lib/eigen_3.2.9/ -I/home/bbales2/stan_math/ -I/home/bbales2/stan_math/lib/cvodes_2.9.0/include --std=c++11 -g -O2 -msse
LFLAGS=-llapack -fopenmp
DEPS=util.hpp mechanics.hpp polybasis.hpp
SOURCES=main.cpp unit_tests1.cpp unit_tests2.cpp stan_test.cpp
CC=g++
#clang++

.PHONY: clean all

all: main unit_tests1 unit_tests2 stan_test

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

depend: .depend

.depend: $(SOURCES)
	rm -f ./.depend
	$(CC) $(CPPFLAGS) -MM $^>>./.depend;

include .depend

main: main.o
	$(CC) $< $(LFLAGS) -o $@

unit_tests1: unit_tests1.o
	$(CC) $< $(LFLAGS) -o $@

unit_tests2: unit_tests2.o
	$(CC) $< $(LFLAGS) -o $@

stan_test: stan_test.o
	$(CC) $< $(LFLAGS) -o $@

clean:
	-rm main.o unit_tests1.o unit_tests2.o main unit_tests1 unit_tests2 stan_test stan_test.o
