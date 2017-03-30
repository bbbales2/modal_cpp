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
  // Rescaling the matrices a little seems to help the solve go faster -- gotta rescale the evecs at the end cause of how LAPACK normalizes then
  //double max = std::min(A.maxCoeff(), B.maxCoeff());

  MatrixXd At = A;// / max;
  MatrixXd Bt = B;// / max;

  il += 1;
  iu += 1;
  
  int itype = 1;
  int N = A.rows();
  int lda = N;
  int ldb = N;
  int ldz = N;
  double abstol = 1e-13;//-11e-13;
  int M = 0;
  eigs.resize(iu - il + 1, 1);
  evecs.resize(N, iu - il + 1);
  std::vector<int> iwork(5 * N);
  std::vector<int> ifail(N);
  int info = 0;
  double zero = 0.0;
  double workQuery = 0.0;
  int lworkQuery = -1;

  //double tmp = omp_get_wtime();

  // First call computes optimal workspace storage size
  dsygvx_(&itype, "V", "I", "U", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, eigs.data(), evecs.data(), &ldz, &workQuery, &lworkQuery, &iwork[0], &ifail[0], &info);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, workQuery, omp_get_wtime() - tmp);

  int lwork = int(workQuery) + 1;
  std::vector<double> work(lwork);

  // Second call actually computes the eigenvalues and eigenvectors!
  dsygvx_(&itype, "V", "I", "U", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, eigs.data(), evecs.data(), &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info);

  //evecs /= sqrt(max);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, work[0], omp_get_wtime() - tmp);
}

// Mechanics solver
//   This takes in the parameters, some lookup tables, and some constants
//   and outputs the resonance modes and all the derivatives of the
//   resonance modes with respect to each parameter
//
//   Input: c11, anisotropic, c44 <- Changing mechanics parameters
//          dp, pv <-- Lookup tables for Rayleigh-Ritz approx to problem
//          nev <-- Number of resonance modes to compute
//   Output:
//          freqs nev x 1 <-- Matrix of resonance modes (directly comparable to data)
//          (optional) dfreqs_dc11, dfreqs_da, dfreqs_dc44 nev x 1 <-- Matrices of derivatives of each resonance mode with respect to each parameter
void mechanics(double c11, double anisotropic, double c44, //Changing parameters
               const dpT& dp, const pvT& pv, unsigned int nevs, // Constants
               VectorXd& freqs,  // Output
               VectorXd& dfreqs_dc11,
               VectorXd& dfreqs_da,
               VectorXd& dfreqs_dc44) { // Derivatives
  double c12 = -(c44 * 2.0 / anisotropic - c11);

  MatrixXd C(6, 6);

  C << c11, c12, c12, 0, 0, 0,
    c12, c11, c12, 0, 0, 0,
    c12, c12, c11, 0, 0, 0,
    0, 0, 0, c44, 0, 0,
    0, 0, 0, 0, c44, 0,
    0, 0, 0, 0, 0, c44;

  MatrixXd dCdc11(6, 6);

  dCdc11 << 1.0, 1.0, 1.0, 0, 0, 0,
    1.0, 1.0, 1.0, 0, 0, 0,
    1.0, 1.0, 1.0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;
  
  MatrixXd dCda(6, 6);

  double dc12da = c44 * 2.0 / (anisotropic * anisotropic);

  dCda << 0.0, dc12da, dc12da, 0, 0, 0,
    dc12da, 0.0, dc12da, 0, 0, 0,
    dc12da, dc12da, 0.0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

  MatrixXd dCdc44(6, 6);

  double dc12dc44 = -2.0 / anisotropic;

  dCdc44 << 0, dc12dc44, dc12dc44, 0, 0, 0,
    dc12dc44, 0, dc12dc44, 0, 0, 0,
    dc12dc44, dc12dc44, 0, 0, 0, 0,
    0, 0, 0, 1.0, 0, 0,
    0, 0, 0, 0, 1.0, 0,
    0, 0, 0, 0, 0, 1.0;

  MatrixXd K, M, _;
  MatrixXd dKdc11, dKda, dKdc44;

  double tmp = omp_get_wtime();
  buildKM(C, dp, pv, K, M);
  buildKM(dCdc11, dp, pv, dKdc11, _);
  buildKM(dCda, dp, pv, dKda, _);
  buildKM(dCdc44, dp, pv, dKdc44, _);

  if(DEBUG)
    printf("buildKM %f\n", omp_get_wtime() - tmp);

  VectorXd eigs;
  MatrixXd evecs;
  
  tmp = omp_get_wtime();
  eigSolve(K, M, 6, 6 + nevs - 1, eigs, evecs);

  int N = K.rows();
  
  if(DEBUG)
    printf("eigSolve %f\n", omp_get_wtime() - tmp);
  
  freqs.resize(nevs);

  dfreqs_dc11.resize(nevs);

  dfreqs_da.resize(nevs);

  dfreqs_dc44.resize(nevs);

  for(int i = 0; i < eigs.rows(); i++) {
    dfreqs_dc11(i, 0) = (evecs.block(0, i, N, 1).transpose() * dKdc11 * evecs.block(0, i, N, 1))(0, 0);
    dfreqs_da(i, 0) = (evecs.block(0, i, N, 1).transpose() * dKda * evecs.block(0, i, N, 1))(0, 0);
    dfreqs_dc44(i, 0) = (evecs.block(0, i, N, 1).transpose() * dKdc44 * evecs.block(0, i, N, 1))(0, 0);
  }

  for(int i = 0; i < eigs.rows(); i++) {
    freqs(i, 0) = sqrt(eigs(i, 0) * 1.0e11) / (M_PI * 2000.0);
    double dfde = 0.5e11 / (sqrt(eigs(i, 0) * 1.0e11) * M_PI * 2000.0);

    dfreqs_dc11(i, 0) *= dfde;
    dfreqs_da(i, 0) *= dfde;
    dfreqs_dc44(i, 0) *= dfde;
  }
}

#endif
