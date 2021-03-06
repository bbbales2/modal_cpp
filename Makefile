CPPFLAGS=-Ispectra/include/Spectra -I../../stan/lib/stan_math/lib/tbb_2019_U8/include -I../../src -I../../stan/src -I../../lib/rapidjson_1.1.0/ -I../../stan/lib/stan_math/ -I../../stan/lib/stan_math/lib/eigen_3.3.3 -I../../stan/lib/stan_math/lib/boost_1.72.0 -I../../stan/lib/stan_math/lib/sundials_5.1.0/include --std=c++1y -g -O3 -msse -pthread
LFLAGS=-llapack -Wl,-L,"/home/bbales2/cmdstan-rus/stan/lib/stan_math/lib/tbb" -Wl,-rpath,"/home/bbales2/cmdstan-rus/stan/lib/stan_math/lib/tbb" -pthread ../../stan/lib/stan_math/lib/tbb/libtbb.so.2
DEPS=util.hpp mechanics.hpp stan_mech.hpp polybasis.hpp
SOURCES=stan_test.cpp stan_test_2.cpp bilayer_test.cpp stan_test_r.cpp cu2qu.cpp
CC=g++
LINK=g++

.PHONY: clean all

all: stan_test stan_test_2 bilayer_test stan_test_r

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

depend: .depend

.depend: $(SOURCES)
	rm -f ./.depend
	g++ $(CPPFLAGS) -MM $^>>./.depend;

include .depend

stan_test: stan_test.o
	$(LINK) $< $(LFLAGS) -o $@

stan_test_2: stan_test_2.o
	$(LINK) $< $(LFLAGS) -o $@

bilayer_test: bilayer_test.o
	$(LINK) $< $(LFLAGS) -o $@

stan_test_r: stan_test_r.o
	$(LINK) $< $(LFLAGS) -o $@

cu2qu: cu2qu.o
	$(LINK) $< $(LFLAGS) -o $@

clean:
	-rm main.o stan_test stan_test.o stan_test_2 stan_test_2.o bilayer_test bilayer_test.o stan_test_r stan_test_r.o cu2qu cu2qu.o
