#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <random>

#include "util.hpp"
#include "polybasis.hpp"
#include "mechanics.hpp"

const double X = 0.007753,
  Y = 0.009057,
  Z = 0.013199;

const double density = 4401.695921;
const double sigma = 0.5; // Not estimating sigma for simple case!!

const double dt = 1e-4;
const double L = 50;

int P = 10;

int N = 30;
ArrayXd data(N, 1);

Eigen::Tensor<double, 4> *dp;
Eigen::Tensor<double, 2> *pv;

std::default_random_engine generator;
std::normal_distribution<double> normal(0.0, 1.0);
std::uniform_real_distribution<double> uniform(0.0, 1.0);

void UgradU(ArrayXd p, // n params
            double *logp, // Log liklihood
            ArrayXd *dlogp) { // Dlog likelihood Dparams
  MatrixXd *freqs, *dfreqs_dc11, *dfreqs_da, *dfreqs_dc44;

  mechanics(p(0, 0), p(1, 0), p(2, 0), // Params
            dp, pv, density, N, // Ref data
            &freqs, // Output
            &dfreqs_dc11, &dfreqs_da, &dfreqs_dc44); // Gradients

  ArrayXd diff = (freqs->array() - data).eval();
  *logp = 0.5 * (diff * diff).sum() / (sigma * sigma);
  (*dlogp)(0, 0) = (dfreqs_dc11->array() * diff).sum() / (sigma * sigma);
  (*dlogp)(1, 0) = (dfreqs_da->array() * diff).sum() / (sigma * sigma);
  (*dlogp)(2, 0) = (dfreqs_dc44->array() * diff).sum() / (sigma * sigma);
  
  delete freqs;
  delete dfreqs_dc11;
  delete dfreqs_da;
  delete dfreqs_dc44;
}

ArrayXd sample(ArrayXd cq) {
  ArrayXd q = cq;

  ArrayXd p(3, 1);
  for(int i = 0; i < 3; i++)
    p(i, 0) = normal(generator);
  
  ArrayXd cp = p;

  ArrayXd dlogp(3, 1);
  double logp;

  UgradU(q, &logp, &dlogp);

  double cU = logp;
  
  p -= dt * dlogp / 2.0;

  for(int i = 0; i < L; i++) {
    q += dt * p;

    UgradU(q, &logp, &dlogp);

    //std::cout << q << std::endl;

    if(i != L - 1)
      p -= dt * dlogp;
  }

  UgradU(q, &logp, &dlogp);

  p -= dt * dlogp / 2.0;

  p = -p;

  double cK = (cp * cp).sum() / 2.0;
  double pU = logp;
  double pK = (p * p).sum() / 2.0;

  //std::cout << "cu" << cU << std::endl;
  //std::cout << "pu" << pU << std::endl;
  //std::cout << "ck" << cK << std::endl;
  //std::cout << "pk" << pK << std::endl;
  
  if(uniform(generator) < exp(cU - pU + cK - pK)) {
    //std::cout << "accepted" << std::endl << std::endl;
    return q;
  } else {
    //std::cout << "rejected" << std::endl << std::endl;
    return cq;
  }
}

int main(int argc, char **argv) {
  // Load up experimental data
  data << 109.076, 136.503, 144.899, 184.926, 188.476, 195.562,
    199.246, 208.46 , 231.22 , 232.63 , 239.057, 241.684,
    242.159, 249.891, 266.285, 272.672, 285.217, 285.67 ,
    288.796, 296.976, 301.101, 303.024, 305.115, 305.827,
    306.939, 310.428, 318.   , 319.457, 322.249, 323.464;

  buildBasis(P, X, Y, Z, &dp, &pv);

  double logp;
  int S = 1000;

  ArrayXd q(3, 1);
  ArrayXd qs(3, N);
  q << 1.685, 1.0, 0.446;

  for(int i = 0; i < S; i++) {
    qs.block(0, i, 3, 1) = sample(q);

    std::cout << qs.block(0, i, 3, 1).transpose() << std::endl;
  }

  /*ArrayXd p(3, 1);
  for(int i = 0; i < 3; i++)
    p(i, 0) = normal(generator);
  
  ArrayXd dlogp(3, 1);

  UgradU(p, &logp, &dlogp);*/

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
