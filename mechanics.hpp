#ifndef mechanics_hpp_
#define mechanics_hpp_

#include <SymEigsShiftSolver.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "util.hpp"

// Lapack dense symmetric eigensolve
extern "C" void dsyevx_(const char *jobz,
                        const char *range,
                        const char *uplo,
                        int *N,
                        double *A,
                        int *lda,
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
void eigSolve(const MatrixXd &A, int il, int iu, VectorXd& eigs, MatrixXd& evecs) {
  MatrixXd At = A;
  
  il += 1;
  iu += 1;
  
  int N = A.rows();
  int lda = N;
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

  //double tmp = omp_get_wtime();
  
  // First call computes optimal workspace storage size
  dsyevx_("V", "I", "U", &N, At.data(), &lda, &zero, &zero, &il, &iu, &abstol, &M, W.data(), evecs.data(), &ldz, &workQuery, &lworkQuery, &iwork[0], &ifail[0], &info);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, workQuery, omp_get_wtime() - tmp);

  int lwork = int(workQuery) + 1;
  std::vector<double> work(lwork);
  
  //tmp = omp_get_wtime();
  // Second call actually computes the eigenvalues and eigenvectors!
  dsyevx_("V", "I", "U", &N, At.data(), &lda, &zero, &zero, &il, &iu, &abstol, &M, W.data(), evecs.data(), &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info);

  eigs = W.segment(0, iu - il + 1);
  //evecs /= sqrt(max);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, work[0], omp_get_wtime() - tmp);
}

// Mechanics solver
//   This takes in the parameters, some lookup tables, and some constants
//   and outputs the resonance modes and all the derivatives of the
//   resonance modes with respect to each parameter
//
//   Input: C <- Elastic constants
//          lookup <-- Lookup tables for Rayleigh-Ritz approx to problem
//          nev <-- Number of resonance modes to compute
//   Output:
//          freqs nev x 1 <-- Matrix of resonance modes (directly comparable to data)
//          dfreqsdCij <-- Derivatives of all output frequencies (columns) with respect to all
//                          parameters of C
void mechanics(const VectorXd& C, //Changing parameters
               const VectorXd& lookup, int nevs, // Constants
               VectorXd& freqs,  // Output
               Matrix<double, Dynamic, Dynamic>& dfreqsdCij) { // Derivatives
  //double tmp = omp_get_wtime();

  int L = 1;

  for(L = 1; L < lookup.size(); L++) {
    if(3 * 3 * L * L * C.size() == lookup.size()) {
      break;
    }
  }
  
  if(L == lookup.size())
    throw std::runtime_error("Solve for L in 3 * 3 * L * L * C.size() == lookup.size() failed");

  MatrixXd K = MatrixXd::Zero(3 * L, 3 * L);

  for(int i = 0; i < C.size(); i++) {
    Map< const Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[i * L * L * 3 * 3], 3 * L, 3 * L);

    for(int l = 0; l < K.size(); l++)
      K(l) += dKdcij(l) * C(i);
  }

  //tmp = omp_get_wtime();
  Spectra::DenseSymShiftSolve<double> op(K);
  Spectra::SymEigsShiftSolver<double, Spectra::LARGEST_MAGN, Spectra::DenseSymShiftSolve<double> > esolve(&op, 6 + nevs, 12 + 2 * nevs, 1e-4);

  // Initialize and compute
  esolve.init();
  int nconv = esolve.compute();
  //printf("Eigensolve: %f\n", omp_get_wtime() - tmp);

  // Retrieve results
  if(esolve.info() != Spectra::SUCCESSFUL) {
    throw std::runtime_error("Eigenvalue solve failed!");
  }

  VectorXd eigs(nevs);
  MatrixXd evecsr = esolve.eigenvectors();
  MatrixXd evecs(esolve.eigenvectors().rows(), nevs);

  for(int i = 0; i < 6; i++) {
    if(esolve.eigenvalues()(6 + nevs - i - 1) > 1e-6) {
      std::cout << "Eigenvalue " << i << " is " << esolve.eigenvalues()(6 + nevs - i - 1) << " (should be near zero -- tolerance is 1e-6)" <<std::endl;
      throw std::runtime_error("Less than six zero eigenvalues. Something has gone wrong");
    }
  }
  
  for(int i = 0; i < nevs; i++) {
    eigs(i) = esolve.eigenvalues()(nevs - i - 1);
    for(int j = 0; j < evecs.rows(); j++) {
      evecs(j, i) = evecsr(j, nevs - i - 1);
    }
  }
  
  //VectorXd eigs;
  //MatrixXd evecs;
  //eigSolve(K, 6, 6 + nevs - 1, eigs, evecs);
  //tmp = omp_get_wtime();

  //std::cout << "its: " << esolve.num_iterations() << std::endl;
  //std::cout << "ops: " << esolve.num_operations() << std::endl;
  //std::cout << esolve.eigenvalues().transpose() << std::endl;

  //std::cout << "====" << std::endl;
  
  //SelfAdjointEigenSolver<MatrixXd> es(K);

  //for(int i = 0; i < 6 + nevs; i++)
  //  std::cout << es.eigenvalues()(i) << std::endl;

  int N = K.rows();

  freqs.resize(nevs);

  dfreqsdCij.resize(nevs, C.size());

  std::vector< MatrixXd > dKdcij_evecs;

  for(int i = 0; i < C.size(); i++) {
    Map< const Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[i * L * L * 3 * 3], 3 * L, 3 * L);

    dKdcij_evecs.push_back(dKdcij * evecs);
  }

  for(int k = 0; k < nevs; k++) {
    freqs(k) = sqrt(eigs(k) * 1.0e11) / (M_PI * 2000.0);
    double dfde = 0.5e11 / (sqrt(eigs(k) * 1.0e11) * M_PI * 2000.0);

    VectorXd evec = evecs.block(0, k, N, 1).eval();
    RowVectorXd evecT = evec.transpose().eval();

    for(int i = 0; i < C.size(); i++) {
      dfreqsdCij(k, i) = (evecT * dKdcij_evecs[i].block(0, k, N, 1))(0, 0) * dfde;
    }
  }
  
  //if(DEBUG)
  //printf("Output prep: %f\n", omp_get_wtime() - tmp);
}

#endif
