CPPFLAGS=-fopenmp -I/home/bbales2/stan_math/lib/boost_1.62.0 -I/home/bbales2/stan_math/lib/eigen_3.2.9/ -I/home/bbales2/stan_math/ -I/home/bbales2/stan_math/lib/cvodes_2.9.0/include --std=c++11 -g -msse
LFLAGS=-llapack -fopenmp
DEPS=util.hpp mechanics.hpp stan_mech.hpp polybasis.hpp
SOURCES=stan_test.cpp stan_test_r.cpp cmsx4.cpp cu2qu.cpp
CC=clang++
LINK=clang++

.PHONY: clean all

all: stan_test stan_test_r

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

depend: .depend

.depend: $(SOURCES)
	rm -f ./.depend
	g++ $(CPPFLAGS) -MM $^>>./.depend;

include .depend

stan_test: stan_test.o
	$(LINK) $< $(LFLAGS) -o $@

stan_test_r: stan_test_r.o
	$(LINK) $< $(LFLAGS) -o $@

cmsx4: cmsx4.o
	$(LINK) $< $(LFLAGS) -o $@

cu2qu: cu2qu.o
	$(LINK) $< $(LFLAGS) -o $@

clean:
	-rm main.o stan_test stan_test.o stan_test_r stan_test_r.o cmsx4 cmsx4.o cu2qu cu2qu.o
