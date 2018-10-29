CPPFLAGS=-fopenmp -Ispectra/include/Spectra -I../../stan/lib/stan_math/lib/boost_1.66.0 -I../../stan/lib/stan_math/lib/eigen_3.3.3/ -I../../stan/lib/stan_math/ -I../../stan/lib/stan_math/lib/sundials_3.1.0/include/ --std=c++1y -g -O3 -msse
LFLAGS=-fopenmp -llapack
DEPS=util.hpp mechanics.hpp stan_mech.hpp polybasis.hpp
SOURCES=stan_test.cpp bilayer_test.cpp stan_test_r.cpp cu2qu.cpp
CC=clang++
LINK=clang++

.PHONY: clean all

all: stan_test bilayer_test stan_test_r

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

depend: .depend

.depend: $(SOURCES)
	rm -f ./.depend
	g++ $(CPPFLAGS) -MM $^>>./.depend;

include .depend

stan_test: stan_test.o
	$(LINK) $< $(LFLAGS) -o $@

bilayer_test: bilayer_test.o
	$(LINK) $< $(LFLAGS) -o $@

stan_test_r: stan_test_r.o
	$(LINK) $< $(LFLAGS) -o $@

cu2qu: cu2qu.o
	$(LINK) $< $(LFLAGS) -o $@

clean:
	-rm main.o stan_test stan_test.o bilayer_test bilayer_test.o stan_test_r stan_test_r.o cu2qu cu2qu.o
