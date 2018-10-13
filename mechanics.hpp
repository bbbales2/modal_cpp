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
  double tmp = omp_get_wtime();

  int L = 1;

  for(L = 1; L < lookup.size(); L++) {
    if(3 * 3 * L * L * C.size() == lookup.size()) {
      break;
    }
  }
  
  if(L == lookup.size())
    throw std::runtime_error("Solve for L in 3 * 3 * L * L * C.size() == lookup.size() failed");

  MatrixXf K = MatrixXf::Zero(3 * L, 3 * L);

  int ij = 0;
  for(int i = 0; i < C.size(); i++) {
    Map< const Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[ij * L * L * 3 * 3], 3 * L, 3 * L);

    for(int l = 0; l < K.size(); l++)
      K(l) += dKdcij(l) * C(i);

    ij++;
  }
  printf("Build matrix: %f\n", omp_get_wtime() - tmp);
  
  tmp = omp_get_wtime();
  Spectra::DenseSymShiftSolve<float> op(K);
  Spectra::SymEigsShiftSolver<float, Spectra::LARGEST_MAGN, Spectra::DenseSymShiftSolve<float> > esolve(&op, 6 + nevs, 12 + 2 * nevs, 2.0);

  // Initialize and compute
  esolve.init();
  int nconv = esolve.compute();
  printf("Eigensolve: %f\n", omp_get_wtime() - tmp);

  // Retrieve results
  if(esolve.info() != Spectra::SUCCESSFUL) {
    throw std::runtime_error("Eigenvalue solve failed!");
  }

  VectorXd eigs(nevs);
  MatrixXf evecsr = esolve.eigenvectors();
  MatrixXd evecs(esolve.eigenvectors().rows(), nevs);
  tmp = omp_get_wtime();

  //std::cout << esolve.eigenvalues() << std::endl;

  //std::cout << "====" << std::endl;
  
  //SelfAdjointEigenSolver<MatrixXd> es(K);

  //for(int i = 0; i < 6 + nevs; i++)
  //  std::cout << es.eigenvalues()(i) << std::endl;

  for(int i = 0; i < nevs; i++) {
    eigs(i) = esolve.eigenvalues()(nevs - i - 1);
    for(int j = 0; j < evecs.rows(); j++) {
      evecs(j, i) = evecsr(j, nevs - i - 1);
    }
  }

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
  printf("Output prep: %f\n", omp_get_wtime() - tmp);
}

#endif
