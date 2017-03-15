#include <eigen3/unsupported/Eigen/CXX11/Tensor>
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

int main(char *argc, char **argv)
{
  Eigen::Tensor<double, 2, Eigen::ColMajor> A(2, 2);

  A.setValues({{1.0, 2.0},
               {2.0, 4.0}});

  int ido = 0;
  int N = 2;
  int nev = 2;
  double tol = 1e-5;
  Eigen::Tensor<double, 1> resid(N);
  int ncv = 2;
  Eigen::Tensor<double, 2, Eigen::ColMajor> v(N, ncv);
  int ldv = 2;
  Eigen::Tensor<int, 1> iparam(11);
  iparam(0) = 0;
  iparam(2) = 100;
  iparam(6) = 1;
  Eigen::Tensor<int, 1> ipntr(11);
  Eigen::Tensor<double, 1> workd(3 * N);
  int lworkl = ncv * ncv + 8 * ncv;
  Eigen::Tensor<double, 1> workl(lworkl);
  int info = 0;

  dsaupd_(&ido, "I", &N, "SM", &nev, &tol, resid.data(), &ncv, v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(), workl.data(), &lworkl, &info);

  auto t = A * A;

  printf("%f", t(0, 0));
  
  //workd.data()[ipntr[0] - 1], workd.data()[iptr[1] - 1];
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
