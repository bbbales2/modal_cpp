#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <stdio.h>

#include "polybasis.hpp"

int main(char *argc, char **argv)
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
