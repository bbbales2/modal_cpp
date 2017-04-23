#%%

import numpy
import scipy
import scipy.linalg
import os
import sys
import random
import json

sys.path.append('/home/bbales2/modal/')

import pyximport
pyximport.install(reload_support = True)
import polybasisqu

from rotations import inv_rotations

Ps = [(10, 286), (12, 455)]
Ns = range(20, 31)
J = 4

def func(P, N, c11, c12, c44, density, w, x, y, z, X, Y, Z):
    #$c12 = -(c44 * 2.0 / anisotropic - c11)

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(P, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, w, x, y, z)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K.astype('float'), M.astype('float'), eigvals = (6, 6 + N - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

# Data from https://arxiv.org/pdf/1605.09237.pdf and http://www.coolmagnetman.com/magconda.htm and https://en.wikipedia.org/wiki/Nickel
cs = [[1.684, 1.214, 0.754, 8960.0], [1.073, 0.6008, 0.283, 2700.0], [2.53, 1.52, 1.24, 8908.0]]
qsub = """#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00

export OMP_NUM_THREADS=1

cd $PBS_O_WORKDIR

/home/bbales2/modal_cpp/models/cubic_w_rotations sample algorithm=hmc engine=nuts max_depth=7 num_samples=1000 num_warmup=1000 save_warmup=1 id=$(echo $PBS_JOBID | sed -e 's/[^0-9]//g') data file=input.dat output file=$PBS_JOBID.csv refresh=1
"""

tmp = """
P <- {0}
N <- {1}
X <- {2}
Y <- {3}
Z <- {4}
density <- {5}
L <- {6}
y <- c({7})
"""

for i in range(50):
    sigma = numpy.random.uniform(0.1, 0.4)

    X = numpy.random.uniform(0.005, 0.02)
    Y = numpy.random.uniform(0.005, 0.02)
    Z = numpy.random.uniform(0.005, 0.02)

    c1 = numpy.random.rand()
    c2 = numpy.random.rand()
    c3 = numpy.random.rand()

    w, x, y, z = inv_rotations.cu2qu([c1, c2, c3])

    c11, c12, c44, density = random.choice(cs)

    P, L = random.choice(Ps)
    N = random.choice(Ns)

    noise = numpy.random.randn(N) * sigma

    freqs = func(P, N, c11, c12, c44, density, w, x, y, z, X, Y, Z) + noise

    os.mkdir(str(i))

    with open("{0}/input.dat".format(i), "w") as f:
        f.write(tmp.format(P, N, X, Y, Z, density, L, ", ".join([str(freq) for freq in freqs])))

    with open("{0}/ref.dat".format(i), "w") as f:
        json.dump({ "P" : P,
                    "N" : N,
                    "c11" : c11, "c12" : c12, "c44" : c44,
                    "density" : density,
                    "w" : w, "x" : x, "y" : y, "z" : z,
                    "X" : X, "Y" : Y, "Z" : Z,
                    "freqs" : list(freqs),
                    "noise" : list(noise) }, f)

    with open("{0}/run.qsub".format(i), "w") as f:
        f.write(qsub)

    print "cd {0}".format(i)
    for j in range(J):
        print "qsub run.qsub".format(i)
    print "cd -"