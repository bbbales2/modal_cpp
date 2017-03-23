#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <stdio.h>
#include <cmath>

#include "util.hpp"
#include "polybasis.hpp"
#include "mechanics.hpp"

const double X = 0.007753,
  Y = 0.009057,
  Z = 0.013199;

const double density = 4401.695921;
const double sigma = 0.5; // Not estimating sigma for simple case!!

int P = 10;

int N = 30;
ArrayXd data(N, 1);

Eigen::Tensor<double, 4> *dp;
Eigen::Tensor<double, 2> *pv;

double tol = 1e-4;

void UgradU(ArrayXd p, // n params
            double &logp, // Log liklihood
            ArrayXd *dlogp) { // Dlog likelihood Dparams
  MatrixXd *freqs, *dfreqs_dc11, *dfreqs_da, *dfreqs_dc44;

  mechanics(p(0, 0), p(1, 0), p(2, 0), // Params
            dp, pv, density, N, // Ref data
            &freqs, // Output
            &dfreqs_dc11, &dfreqs_da, &dfreqs_dc44); // Gradients

  ArrayXd diff = (freqs->array() - data).eval();
  logp = 0.5 * (diff * diff).sum() / (sigma * sigma);
  (*dlogp)(0, 0) = (dfreqs_dc11->array() * diff).sum() / (sigma * sigma);
  (*dlogp)(1, 0) = (dfreqs_da->array() * diff).sum() / (sigma * sigma);
  (*dlogp)(2, 0) = (dfreqs_dc44->array() * diff).sum() / (sigma * sigma);
  
  delete freqs;
  delete dfreqs_dc11;
  delete dfreqs_da;
  delete dfreqs_dc44;
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

  ArrayXd p(3, 1), pt(3, 1);
  p << 1.685, 1.0, 0.446;

  ArrayXd dlogp(3, 1);

  UgradU(p, logp, &dlogp);

  if(abs(logp - 61.9090538264) / 61.9090538264 < tol) {
    std::cout << "Log probability passed" << std::endl;
  } else {
    std::cout << "Log probability failed" << std::endl;
  }

  if(abs(dlogp(0, 0) + 868.912090908) / 868.912090908 < tol) {
    std::cout << "dc11 derivative passed" << std::endl;
  } else {
    std::cout << "dc11 derivative failed" << std::endl;
  }

  double logpt, delta = 0.00001;

  pt = p;
  pt(0, 0) = p(0, 0) + delta;
  UgradU(pt, logpt, &dlogp);

  if(abs(((logpt - logp) / delta - dlogp(0, 0)) / dlogp(0, 0)) < tol) {
    std::cout << "dc11 numerical gradient passed" << std::endl;
  } else {
    std::cout << "dc11 numerical gradient failed" << std::endl;
  }
  
  pt = p;
  pt(1, 0) = p(1, 0) + delta;
  UgradU(pt, logpt, &dlogp);

  if(abs(((logpt - logp) / delta - dlogp(1, 0)) / dlogp(0, 0)) < tol) {
    std::cout << "da numerical gradient passed" << std::endl;
  } else {
    std::cout << "da numerical gradient failed" << std::endl;
  }

  pt = p;
  pt(2, 0) = p(2, 0) + delta;
  UgradU(pt, logpt, &dlogp);

  if(abs(((logpt - logp) / delta - dlogp(2, 0)) / dlogp(0, 0)) < tol) {
    std::cout << "dc44 numerical gradient passed" << std::endl;
  } else {
    std::cout << "dc44 numerical gradient failed" << std::endl;
  }

  return 0;
}
