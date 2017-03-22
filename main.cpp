#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <stdio.h>

#include "polybasis.hpp"
#include "mechanics.hpp"

int main(int argc, char **argv)
{
  double X = 0.007753,
    Y = 0.009057,
    Z = 0.013199;

  double c11 = 1.685;
  double anisotropic = 1.0;
  double c44 = 0.446;

  double density = 4401.695921;

  Eigen::Tensor<double, 4> *dp;
  Eigen::Tensor<double, 2> *pv;
  
  double tmp = omp_get_wtime();
  buildBasis(10, X, Y, Z, &dp, &pv);
  printf("buildBasis %f\n", omp_get_wtime() - tmp);

  MatrixXd *eigs, *evecs;

  tmp = omp_get_wtime();
  mechanics(c11, anisotropic, c44, // Params
            dp, pv, density, // Ref data
            &eigs, &evecs); // Output
  printf("mechanics %f\n", omp_get_wtime() - tmp);

  printf("Eigenvalues:\n");
  for(int i = 0; i < eigs->rows(); i++) {
    for(int j = 0; j < eigs->cols(); j++) {
      printf("%e ", (*eigs)(i, j));
    }
    printf("\n");
  }

  return 0;
}

/*int main2(int argc, char **argv) {
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
  MatrixXd *eigs, *evecs;
  eigsD(A, B, 0, 7, &eigs, &evecs);
  printf("Time %f\n", omp_get_wtime() - tmp);
  
  printf("Eigenvalues:\n");
  for(int i = 0; i < eigs.rows(); i++) {
    for(int j = 0; j < eigs.cols(); j++) {
      printf("%e ", eigs(i, j));
    }
    printf("\n");
  }

  //printf("\nEigenvectors:\n");
  //for(int i = 0; i < N; i++) {
  //  for(int j = 0; j < nev; j++) {
  //    printf("%f ", Z(i, j));
  //  }
  //  printf("\n");
  //}
}*/
