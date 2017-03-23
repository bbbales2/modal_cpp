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

double dt = 1e-5;
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
            double *nlogp, // negative Log liklihood
            ArrayXd *dnlogp) { // negative Dlog likelihood Dparams
  MatrixXd *freqs, *dfreqs_dc11, *dfreqs_da, *dfreqs_dc44;

  mechanics(p(0, 0), p(1, 0), p(2, 0), // Params
            dp, pv, density, N, // Ref data
            &freqs, // Output
            &dfreqs_dc11, &dfreqs_da, &dfreqs_dc44); // Gradients

  double sigma = p(3, 0);

  //std::cout << freqs->transpose() << std::endl;
  
  ArrayXd diff = (freqs->array() - data).eval();
  *nlogp = 0.5 * (diff * diff).sum() / (sigma * sigma) + 2 * N * log(sigma);
  (*dnlogp)(0, 0) = (dfreqs_dc11->array() * diff).sum() / (sigma * sigma);
  (*dnlogp)(1, 0) = (dfreqs_da->array() * diff).sum() / (sigma * sigma);
  (*dnlogp)(2, 0) = (dfreqs_dc44->array() * diff).sum() / (sigma * sigma);
  (*dnlogp)(3, 0) = 2 * N / sigma - (diff * diff).sum() / (sigma * sigma * sigma);

  delete freqs;
  delete dfreqs_dc11;
  delete dfreqs_da;
  delete dfreqs_dc44;
}

ArrayXd sample(ArrayXd cq) {
  ArrayXd q = cq;

  ArrayXd p(4, 1);
  for(int i = 0; i < 4; i++)
    p(i, 0) = normal(generator);
  
  ArrayXd cp = p;

  ArrayXd dnlogp(4, 1);
  double nlogp;

  UgradU(q, &nlogp, &dnlogp);

  double cU = nlogp;
  std::cout << q.transpose() << " - " << cU << std::endl;
  
  p -= dt * dnlogp / 2.0;

  for(int i = 0; i < L; i++) {
    q += dt * p;

    UgradU(q, &nlogp, &dnlogp);

    //std::cout << q << std::endl;

    if(i != L - 1)
      p -= dt * dnlogp;
  }

  UgradU(q, &nlogp, &dnlogp);

  p -= dt * dnlogp / 2.0;

  p = -p;

  double cK = (cp * cp).sum() / 2.0;
  double pU = nlogp;
  double pK = (p * p).sum() / 2.0;

  /*std::cout << "cu " << cU << std::endl;
  std::cout << "pu " << pU << std::endl;
  std::cout << "ck " << cK << std::endl;
  std::cout << "pk " << pK << std::endl;
  std::cout << "c - p " << cU - pU + cK - pK << std::endl;*/
  
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

  ArrayXd q(4, 1);
  ArrayXd qs(4, S);
  q << 2.0, 1.0, 1.0, 2.0;

  dt = 1e-5;

  q = sample(q);
  q = sample(q);
  q = sample(q);
  q = sample(q);

  dt = 2e-4;
  
  for(int i = 0; i < S; i++) {
    q = sample(q);
    
    qs.block(0, i, 4, 1) = q;

    std::cout << qs.block(0, i, 4, 1).transpose() << std::endl;
  }

  return 0;
}
