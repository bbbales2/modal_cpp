CPPFLAGS=-fopenmp -I/home/bbales2/stan_math/lib/boost_1.62.0 -I/home/bbales2/stan_math/lib/eigen_3.2.9/ -I/home/bbales2/stan_math/ -I/home/bbales2/stan_math/lib/cvodes_2.9.0/include --std=c++11 -g -msse
LFLAGS=-llapack -fopenmp
DEPS=util.hpp mechanics.hpp polybasis.hpp
SOURCES=main.cpp unit_tests1.cpp unit_tests2.cpp stan_test.cpp stan_input_gen.cpp stan_test_r.cpp
CC=clang++
LINK=clang++

.PHONY: clean all

all: main unit_tests1 unit_tests2 stan_test stan_test_r

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

depend: .depend

.depend: $(SOURCES)
	rm -f ./.depend
	g++ $(CPPFLAGS) -MM $^>>./.depend;

include .depend

main: main.o
	$(LINK) $< $(LFLAGS) -o $@

unit_tests1: unit_tests1.o
	$(LINK) $< $(LFLAGS) -o $@

unit_tests2: unit_tests2.o
	$(LINK) $< $(LFLAGS) -o $@

stan_test: stan_test.o
	$(LINK) $< $(LFLAGS) -o $@

stan_test_r: stan_test_r.o
	$(LINK) $< $(LFLAGS) -o $@

stan_input_gen: stan_input_gen.o
	$(LINK) $< $(LFLAGS) -o $@

clean:
	-rm main.o unit_tests1.o unit_tests2.o main unit_tests1 unit_tests2 stan_test stan_test.o stan_test_r stan_test_r.o
