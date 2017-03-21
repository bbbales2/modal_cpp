#define EIGEN_USE_BLAS

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <stdio.h>

#include "polybasis.hpp"

extern "C" void dsaupd_(int *ido,
                       const char *bmat,
                       int *N,
                       const char *which,
                       int *nev,
                       double *tol,
                       double *resid,
                       int *nvc,
                       double *v,
                       int *ldv,
                       int *IPARAM,
                       int *IPNTR,
                       double *workd,
                       double *workl,
                       int *lworkl,
                       int *info);

/*extern "C" void dgemv_(char *trans,
                       int *M,
                       int *N,
                       double *alpha,
                       double *A,
                       int *lda,
                       double *X,
                       int *incx,
                       double *beta,
                       double *Y,
                       int *incy);*/

extern "C" void dseupd_(int *rvec,
                        const char *howmany,
                        int *select,
                        double *D,
                        double *Z,
                        int *ldz,
                        double *sigma,
                        const char *bmat,
                        int *N,
                        const char *which,
                        int *nev,
                        double *tol,
                        double *resid,
                        int *nvc,
                        double *v,
                        int *ldv,
                        int *IPARAM,
                        int *IPNTR,
                        double *workd,
                        double *workl,
                        int *lworkl,
                        int *info);

MatrixXd eigsD(MatrixXd &A, MatrixXd &B, int nev, double sigma_ = 0.5)
{
  double sigma = sigma_; // I have to reassign sigma from arguments to local variable otherwise the output doesn't get shifted right. Not sure what's happening.

  int N = A.rows();
  
  bool si = true;
  const char *weigs = (si) ? "LA" : "SA";
  const char *type = (si) ? "G" : "I";

  auto Ashift = A;

  for(int i = 0; i < A.rows(); i++)
    for(int j = 0; j < A.cols(); j++)
      Ashift(i, j) -= sigma * B(i, j);

  auto LL = B.llt();
  MatrixXd L = LL.matrixL();
  auto LT = L.transpose().eval();

  auto op = (si) ? Ashift.ldlt() : B.ldlt();
  
  int ido = 0;
  double tol = 1e-5;
  std::vector<double> resid(N);
  int ncv = std::min(nev + 10, N - 1);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> v(N, ncv);
  int ldv = N;
  int iparam[11];
    iparam[0] = 1;

  iparam[2] = 10000;
  iparam[3] = 1;

  if(si)
    iparam[6] = 3;
  else
    iparam[6] = 1;

  int ipntr[11];
  std::vector<double> workd(3 * N);
  int lworkl = ncv * ncv + 8 * ncv;
  std::vector<double> workl(lworkl);

  int info = 0;

  double tmp = omp_get_wtime();

  double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0, t;
  
  while(true) {
    t = omp_get_wtime();
    dsaupd_(&ido, type, &N, weigs, &nev, &tol, &resid[0], &ncv, v.data(), &ldv, iparam, ipntr, &workd[0], &workl[0], &lworkl, &info);
    t1 += omp_get_wtime() - t;
    
    //printf("ido: %d\n", ido);
    //printf("%d %d\n", ipntr[0], ipntr[1]);
    //printf("info: %d\n", info);

    if(info != 0) {
      printf("error %d\n", info);
      exit(-1);
    }
    
    t = omp_get_wtime();
    Eigen::Map<Eigen::MatrixXd> x(&workd[ipntr[0] - 1], N, 1),
      y(&workd[ipntr[1] - 1], N, 1);
    t2 += omp_get_wtime() - t;
 
    if(ido == -1) {
      t = omp_get_wtime();
      if(si) {
        y = op.solve(B * x);
      } else {
        y = L.triangularView<Eigen::Lower>().solve(A * LT.triangularView<Eigen::Upper>().solve(x));
      }
      t3 += omp_get_wtime() - t;
    } else if(ido == 1) {
      t = omp_get_wtime();
      Eigen::Map<Eigen::MatrixXd> z(&workd[ipntr[2] - 1], N, 1);
      t2 += omp_get_wtime() - t;

      t = omp_get_wtime();
      if(si) {
        y = op.solve(z);
      } else {
        y = L.triangularView<Eigen::Lower>().solve(A * LT.triangularView<Eigen::Upper>().solve(x));
      }
      t4 += omp_get_wtime() - t;
    } else if(ido == 2) {
      t = omp_get_wtime();
      y = B * x;
      t5 += omp_get_wtime() - t;
    } else if(ido == 99) {
      break;
    } else {
      printf("ido = %d\n", ido);
      exit(-1);
    }
  }

  printf("dsaupd_ %f\nalloc %f\n-1 %f\n1 %f\n2 %f\n", t1, t2, t3, t4, t5);
  
  int rvec = 1;
  int select[nev];
  for(int i = 0; i < nev; i++)
    select[nev] = 1;
  
  MatrixXd D(nev, 1);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Z(N, nev);
  int ldz = N;
  
  dseupd_(&rvec, "A", select, D.data(), Z.data(), &ldz, &sigma,
          type, &N, weigs, &nev, &tol, &resid[0], &ncv, v.data(), &ldv, iparam, ipntr, &workd[0], &workl[0], &lworkl, &info);

  if(!si)
    Z = LT.triangularView<Eigen::Upper>().solve<Eigen::OnTheLeft>(Z);

  printf("Info %d, %f\n", info, omp_get_wtime() - tmp);
  
  return D;
}

int main2(int argc, char **argv) {
  int N = 10;

  MatrixXd A(N, N);
  MatrixXd B(N, N);

  //Eigen::Tensor<double, 2, Eigen::ColMajor> &A = *new Eigen::Tensor<double, 2, Eigen::ColMajor>(N, N);

  //A.setRandom();

  //for(int i = 0; i < N; i++) {
  //  for(int j = i; j < N; j++) {
  //    A(j, i) = A(i, j);
  //  }
  //}

  A.setZero();
  B.setZero();

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      if(i == j) {
        A(i, j) = i + 1.0;
        B(i, j) = 2.00;
        printf("%d %d %f %f\n", i, j, B(i, j), A(i, j));
      }
    }
  }

  double tmp = omp_get_wtime();
  MatrixXd eigs = eigsD(A, B, 5);
  printf("Time %f\n", omp_get_wtime() - tmp);
  
  printf("Eigenvalues:\n");
  for(int i = 0; i < eigs.rows(); i++) {
    for(int j = 0; j < eigs.cols(); j++) {
      printf("%e ", eigs(i, j));
    }
    printf("\n");
  }

  /*printf("\nEigenvectors:\n");
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < nev; j++) {
      printf("%f ", Z(i, j));
    }
    printf("\n");
    }*/
}

int main(int argc, char **argv)
{
  double X = 0.007753,
    Y = 0.009057,
    Z = 0.013199;

  double c11 = 1.685;
  double anisotropic = 1.0;
  double c44 = 0.446;
  double c12 = -(c44 * 2.0 / anisotropic - c11);

  double density = 4401.695921;

  Eigen::Tensor<double, 4> *dp;
  Eigen::Tensor<double, 2> *pv;
  
  build(10, X, Y, Z, &dp, &pv);

  Eigen::Tensor<double, 2> C(6, 6);

  C.setValues({{c11, c12, c12, 0, 0, 0},
               {c12, c11, c12, 0, 0, 0},
               {c12, c12, c11, 0, 0, 0},
               {0, 0, 0, c44, 0, 0},
               {0, 0, 0, 0, c44, 0},
               {0, 0, 0, 0, 0, c44}});

  MatrixXd *K, *M;
  
  buildKM(C, *dp, *pv, density, &K, &M);

  for(int i = 0; i < M->rows(); i++) {
    for(int j = 0; j < M->cols(); j++) {
      (*K)(i, j) /= 1e-6;
      (*M)(i, j) /= 1e-6;
    }
  }

  double tmp = omp_get_wtime();
  MatrixXd eigs = eigsD(*K, *M, 53);
  printf("Time %f\n", omp_get_wtime() - tmp);

  /*M->setZero();

  for(int i = 0; i < M->rows(); i++) {
    for(int j = 0; j < M->cols(); j++) {
      if(i == j) {
        (*M)(i, j) = 1.0;
      }
    }
    } */ 
  
  printf("Eigenvalues:\n");
  for(int i = 0; i < eigs.rows(); i++) {
    for(int j = 0; j < eigs.cols(); j++) {
      printf("%e ", eigs(i, j));
    }
    printf("\n");
  }

  /*for(int i = 0; i < 10; i++) {
    for(int j = 0; j < 10; j++) {
      printf("%e ", (*K)(i, j));
    }
    printf("\n");
    }*/
  
  return 0;
}
