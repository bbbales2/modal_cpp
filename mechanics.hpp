#ifndef mechanics_hpp_
#define mechanics_hpp_

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "util.hpp"

// Lapack dense symmetric, generalized eigensolve
extern "C" void dsygvx_(int *itype,
                        const char *jobz,
                        const char *range,
                        const char *uplo,
                        int *N,
                        double *A,
                        int *lda,
                        double *B,
                        int *ldb,
                        double *vl,
                        double *vu,
                        int *il,
                        int *iu,
                        double *abstol,
                        int *M,
                        double *W,
                        double *Z,
                        int *ldz,
                        double *work,
                        int *lwork,
                        int *iwork,
                        int *ifail,
                        int *info);

// Wrap up the eigensolver in a C++ function
void eigSolve(const MatrixXd &A, const MatrixXd &B, int il, int iu, VectorXd& eigs, MatrixXd& evecs)
{
  MatrixXd At = A;
  MatrixXd Bt = B;
  
  il += 1;
  iu += 1;
  
  int itype = 1;
  int N = A.rows();
  int lda = N;
  int ldb = N;
  int ldz = N;
  double abstol = 1e-13;//-11e-13;
  int M = 0;
  VectorXd W(N);
  eigs.resize(iu - il + 1);
  evecs.resize(N, iu - il + 1);
  std::vector<int> iwork(5 * N);
  std::vector<int> ifail(N);
  int info = 0;
  double zero = 0.0;
  double workQuery = 0.0;
  int lworkQuery = -1;

  // First call computes optimal workspace storage size
  dsygvx_(&itype, "V", "I", "U", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, W.data(), evecs.data(), &ldz, &workQuery, &lworkQuery, &iwork[0], &ifail[0], &info);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, workQuery, omp_get_wtime() - tmp);

  int lwork = int(workQuery) + 1;
  std::vector<double> work(lwork);

  // Second call actually computes the eigenvalues and eigenvectors!
  dsygvx_(&itype, "V", "I", "U", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, W.data(), evecs.data(), &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info);

  eigs = W.segment(0, iu - il + 1);
  //evecs /= sqrt(max);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, work[0], omp_get_wtime() - tmp);
}

// Mechanics solver
//   This takes in the parameters, some lookup tables, and some constants
//   and outputs the resonance modes and all the derivatives of the
//   resonance modes with respect to each parameter
//
//   Input: C <- Positive definite 6x6 compliance matrix. 21 effective parameters
//          lookup <-- Lookup tables for Rayleigh-Ritz approx to problem
//          nev <-- Number of resonance modes to compute
//   Output:
//          freqs nev x 1 <-- Matrix of resonance modes (directly comparable to data)
//          dfreqsdCij <-- Derivatives of all output frequencies (columns) with respect to all
//                          parameters of C
void mechanics(const Matrix<double, 6, 6>& C, //Changing parameters
               int P, const Matrix<double, Dynamic, 1>& lookup, int nevs, // Constants
               VectorXd& freqs,  // Output
               Matrix<double, Dynamic, 21>& dfreqsdCij) { // Derivatives
  int L = (P + 1) * (P + 2) * (P + 3) / 6;

  //double tmp = omp_get_wtime();

  MatrixXd K, M;
  
  buildKM(P, C, lookup, K, M);

  //if(DEBUG)
  //  printf("buildKM %f\n", omp_get_wtime() - tmp);

  VectorXd eigs;
  MatrixXd evecs;
  
  //tmp = omp_get_wtime();
  eigSolve(K, M, 6, 6 + nevs - 1, eigs, evecs);
  
  //if(DEBUG)
  //  printf("eigSolve %f\n", omp_get_wtime() - tmp);

  int N = K.rows();
  
  freqs.resize(nevs);

  dfreqsdCij.resize(nevs, 21);

  std::vector< MatrixXd > dKdcij_evecs;

  //tmp = omp_get_wtime();
  for(int ij = 0; ij < 21; ij++) {
    Map< const Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[L * L * 3 * 3 + L * L + ij * L * L * 3 * 3], 3 * L, 3 * L);

    dKdcij_evecs.push_back(dKdcij * evecs);
  }
  
  for(int i = 0; i < nevs; i++) {
    freqs(i) = sqrt(eigs(i, 0) * 1.0e11) / (M_PI * 2000.0);
    double dfde = 0.5e11 / (sqrt(eigs(i, 0) * 1.0e11) * M_PI * 2000.0);

    VectorXd evec = evecs.block(0, i, N, 1).eval();
    RowVectorXd evecT = evec.transpose().eval();

    for(int ij = 0; ij < 21; ij++)
      dfreqsdCij(i, ij) = (evecT * dKdcij_evecs[ij].block(0, i, N, 1))(0, 0) * dfde;
  }
  
  //if(DEBUG)
  //  printf("Output prep: %f\n", omp_get_wtime() - tmp);
}

#endif
