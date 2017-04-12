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
  eigs.resize(iu - il + 1, 1);
  evecs.resize(N, iu - il + 1);
  std::vector<int> iwork(5 * N);
  std::vector<int> ifail(N);
  int info = 0;
  double zero = 0.0;
  double workQuery = 0.0;
  int lworkQuery = -1;

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

// Lapack dense symmetric, generalized eigensolve
extern "C" void ssygvx_(int *itype,
                        const char *jobz,
                        const char *range,
                        const char *uplo,
                        int *N,
                        float *A,
                        int *lda,
                        float *B,
                        int *ldb,
                        float *vl,
                        float *vu,
                        int *il,
                        int *iu,
                        float *abstol,
                        int *M,
                        float *W,
                        float *Z,
                        int *ldz,
                        float *work,
                        int *lwork,
                        int *iwork,
                        int *ifail,
                        int *info);

// Wrap up the eigensolver in a C++ function
void eigSolvef(const MatrixXd &A, const MatrixXd &B, int il, int iu, VectorXd& eigs_, MatrixXd& evecs_)
{
  MatrixXf At = A.cast<float>();
  MatrixXf Bt = B.cast<float>();

  VectorXf eigs;
  MatrixXf evecs;
  
  il += 1;
  iu += 1;
  
  int itype = 1;
  int N = A.rows();
  int lda = N;
  int ldb = N;
  int ldz = N;
  float abstol = 1e-7;//-11e-13;
  int M = 0;
  eigs.resize(iu - il + 1, 1);
  evecs.resize(N, iu - il + 1);
  std::vector<int> iwork(5 * N);
  std::vector<int> ifail(N);
  int info = 0;
  float zero = 0.0;
  float workQuery = 0.0;
  int lworkQuery = -1;

  float tmp = omp_get_wtime();
  // First call computes optimal workspace storage size
  ssygvx_(&itype, "V", "I", "U", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, eigs.data(), evecs.data(), &ldz, &workQuery, &lworkQuery, &iwork[0], &ifail[0], &info);

  printf("Info %d, M %d, opt %f, %f\n", info, M, workQuery, omp_get_wtime() - tmp);

  int lwork = int(workQuery) + 1;
  std::vector<float> work(lwork);

  tmp = omp_get_wtime();
  // Second call actually computes the eigenvalues and eigenvectors!
  ssygvx_(&itype, "V", "I", "U", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, eigs.data(), evecs.data(), &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info);

  eigs_ = eigs.cast<double>();
  evecs_ = evecs.cast<double>();
  
  //evecs /= sqrt(max);

  printf("Info %d, M %d, opt %f, %f\n", info, M, work[0], omp_get_wtime() - tmp);
}

extern "C" void dsytrd_(const char* uplo,
                        const int *N,
                        double *A,
                        const int *lda,
                        double *D,
                        double *E,
                        double *tau,
                        double *work,
                        const int *lwork,
                        int *info);

void dsytrd(const MatrixXd &A, VectorXd& D, VectorXd& E) {
  MatrixXd At = A;

  int N = A.rows();

  D.resize(N);
  E.resize(N - 1);

  VectorXd tau(N - 1);
  double lwork_;
  int lwork = -1;
  int info;

  dsytrd_("U", &N, At.data(), &N, D.data(), E.data(), tau.data(), &lwork_, &lwork, &info);

  lwork = int(lwork_) + 1;
  VectorXd work(lwork);

  dsytrd_("U", &N, At.data(), &N, D.data(), E.data(), tau.data(), work.data(), &lwork, &info);
}

extern "C" void dstegr_(const char *jobz,
                        const char *range,
                        const int *N,
                        double *D,
                        double *E,
                        const double *vl,
                        const double *vu,
                        const int *il,
                        const int *iu,
                        const double *abstol,
                        int *M,
                        double *W,
                        double *Z,
                        const int *ldz,
                        const int *isuppz,
                        double *work,
                        int *lwork,
                        int *iwork,
                        int *liwork,
                        int *info);

// Wrap up the eigensolver in a C++ function
void dstegr(const VectorXd &D, const VectorXd &E, int il, int iu, VectorXd& eigs, MatrixXd& evecs)
{
  VectorXd Dt = D;
  
  il += 1;
  iu += 1;
  
  int itype = 1;
  int N = D.size();
  VectorXd Et(N);
  for(int i = 0; i < N - 1; i++)
    Et(i) = E(i);
  
  int ldz = N;
  double abstol = 1e-13;//-11e-13;
  int M = 0;
  eigs.resize(iu - il + 1, 1);
  evecs.resize(N, iu - il + 1);
  VectorXd eigs_(N);
  std::vector<int> isuppz(2 * (iu - il + 1));
  int lwork = 18 * N + 1;
  std::vector<double> work(lwork);
  int liwork = 10 * N + 1;
  std::vector<int> iwork(liwork);
  int info = 0;
  double zero = 0.0;
  double workQuery = 0.0;
  int lworkQuery = -1;

  double tmp = omp_get_wtime();
  
  dstegr_("V", "I", &N, Dt.data(), Et.data(), &zero, &zero, &il, &iu, &abstol, &M, eigs_.data(), evecs.data(), &N, isuppz.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

  //evecs /= sqrt(max);

  for(int i = 0; i < iu - il + 1; i++)
    eigs(i) = eigs_(i);

  printf("Info %d, M %d, opt %f, %f\n", info, M, work[0], omp_get_wtime() - tmp);
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
void mechanics(const Matrix<double, 6, 6>& C, //Changing parameters
               const Matrix<double, Dynamic, 1>& lookup, int L, int nevs, // Constants
               VectorXd& freqs,  // Output
               Matrix<double, Dynamic, 21>& dfreqsdCij) { // Derivatives
  double tmp = omp_get_wtime();

  MatrixXd K, M, _;
  
  buildKM(C, lookup, L, K, M);

  if(DEBUG)
    printf("buildKM %f\n", omp_get_wtime() - tmp);

  VectorXd eigs;
  MatrixXd evecs;

  tmp = omp_get_wtime();
  VectorXd D, E;

  auto LL = M.llt();
  MatrixXd L_ = LL.matrixL();
  auto LT = L_.transpose().eval();
  printf("llt %f\n", omp_get_wtime() - tmp);

  tmp = omp_get_wtime();
  auto LLA = L_.triangularView<Eigen::Lower>().solve((L_.triangularView<Eigen::Lower>().solve(K.transpose())).transpose());
  if(DEBUG)
    printf("Lsolveprep %f\n", omp_get_wtime() - tmp);
  
  tmp = omp_get_wtime();
  dsytrd(LLA, D, E);
  dstegr(D, E, 6, 6 + nevs - 1, eigs, evecs);
  
  if(DEBUG)
    printf("tridiag + eigs %f\n", omp_get_wtime() - tmp);

  tmp = omp_get_wtime();
  evecs = LT.triangularView<Eigen::Upper>().solve(evecs);

  if(DEBUG)
    printf("evecs solve %f\n", omp_get_wtime() - tmp);

  tmp = omp_get_wtime();
  //eigSolve(K, M, 6, 6 + nevs - 1, eigs, evecs);
  
  if(DEBUG)
    printf("eigSolve %f\n", omp_get_wtime() - tmp);

  int N = K.rows();
  
  freqs.resize(nevs);

  dfreqsdCij.resize(nevs, 21);

  std::vector< MatrixXd > dKdcij_evecs(21);

  tmp = omp_get_wtime();
  for(int ij = 0; ij < 21; ij++) {
    Map< const Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[L * L * 3 * 3 + L * L + ij * L * L * 3 * 3], 3 * L, 3 * L);
    
    dKdcij_evecs[ij] = dKdcij * evecs;
  }
  
  for(int i = 0; i < nevs; i++) {
    freqs(i) = sqrt(eigs(i, 0) * 1.0e11) / (M_PI * 2000.0);
    double dfde = 0.5e11 / (sqrt(eigs(i, 0) * 1.0e11) * M_PI * 2000.0);

    VectorXd evec = evecs.block(0, i, N, 1).eval();
    RowVectorXd evecT = evec.transpose().eval();

    for(int ij = 0; ij < 21; ij++)
      dfreqsdCij(i, ij) = (evecT * dKdcij_evecs[ij].block(0, i, N, 1))(0, 0) * dfde;
  }
  
  if(DEBUG)
    printf("Output prep: %f\n", omp_get_wtime() - tmp);
}

#endif
