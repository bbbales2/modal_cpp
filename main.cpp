#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Core>
#include <stdio.h>

#include "polybasis.hpp"

extern "C" void dsaupd_(int *ido,
                       char *bmat,
                       int *N,
                       char *which,
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

extern "C" void dgemv_(char *trans,
                       int *M,
                       int *N,
                       double *alpha,
                       double *A,
                       int *lda,
                       double *X,
                       int *incx,
                       double *beta,
                       double *Y,
                       int *incy);

extern "C" void dseupd_(int *rvec,
                        char *howmany,
                        int *select,
                        double *D,
                        double *Z,
                        int *ldz,
                        double *sigma,
                        char *bmat,
                        int *N,
                        char *which,
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

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

int main(char *argc, char **argv)
{
  int N = 20;

  MatrixXd A(N, N);
  auto LUA = A.lu();
  
  //Eigen::Tensor<double, 2, Eigen::ColMajor> &A = *new Eigen::Tensor<double, 2, Eigen::ColMajor>(N, N);

  A.setRandom();

  for(int i = 0; i < N; i++) {
    for(int j = i; j < N; j++) {
      A(j, i) = A(i, j);
    }
  }
  
  int ido = 0;
  int nev = 5;
  double tol = -1e-5;
  std::vector<double> resid(N);
  int ncv = 20;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> v(N, ncv);
  int ldv = N;
  int iparam[11];
  iparam[0] = 1;
  iparam[2] = 1000;
  iparam[3] = 1;
  iparam[6] = 1;
  int ipntr[11];
  std::vector<double> workd(3 * N);
  int lworkl = ncv * ncv + 8 * ncv;
  std::vector<double> workl(lworkl);

  int info = 0;

  while(true) {
    dsaupd_(&ido, "I", &N, "SM", &nev, &tol, &resid[0], &ncv, v.data(), &ldv, iparam, ipntr, &workd[0], &workl[0], &lworkl, &info);
    
    //printf("ido: %d\n", ido);
    //printf("%d %d\n", ipntr[0], ipntr[1]);
    //printf("info: %d\n", info);

    if(ido == 1 || ido == -1) {
      double alpha = 1.0;
      double beta = 0.0;
      int one = 1;

      Eigen::Map<Eigen::MatrixXd> x(&workd[ipntr[0] - 1], N, 1),
        y(&workd[ipntr[1] - 1], N, 1);
      
      y = A * x;
      
      //dgemv_("N", &N, &N, &alpha, A.data(), &N, &workd[ipntr[0] - 1], &one, &beta, &workd[ipntr[1] - 1], &one);
    } else if(ido == 99) {
      break;
    } else {
      printf("ido = %d\n", ido);
      return -1;
    }
  }

  int rvec = 1;
  int select[nev];
  for(int i = 0; i < nev; i++)
    select[nev] = 1;
  std::vector<double> D(nev);
  Eigen::Tensor<double, 2, Eigen::ColMajor> &Z = *new Eigen::Tensor<double, 2, Eigen::ColMajor>(N, nev);
  int ldz = N;
  double sigma = 0.0;
  
  dseupd_(&rvec, "A", select, &D[0], Z.data(), &ldz, &sigma,
          "I", &N, "SM", &nev, &tol, &resid[0], &ncv, v.data(), &ldv, iparam, ipntr, &workd[0], &workl[0], &lworkl, &info);

  printf("Info %d\n", info);
  
  printf("Eigenvalues:\n");
  for(int i = 0; i < nev; i++) {
    printf("%f\n", D[i]);
  }

  /*printf("\nEigenvectors:\n");
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < nev; j++) {
      printf("%f ", Z(i, j));
    }
    printf("\n");
    }*/
}

int main2(char *argc, char **argv)
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

  Eigen::Tensor<double, 2> *K, *M;
  
  buildKM(C, *dp, *pv, density, &K, &M);

  for(int i = 0; i < 10; i++) {
    for(int j = 0; j < 10; j++) {
      printf("%e ", (*K)(i, j));
    }
    printf("\n");
  }
  
  return 0;
}
