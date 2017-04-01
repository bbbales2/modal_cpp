#include <stan/math.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

#include "util.hpp"
#include "polybasis.hpp"
#include "mechanics.hpp"

const double X = 0.007753,
  Y = 0.009057,
  Z = 0.013199;

const double density = 4401.695921;

int P = 10;

int N = 30;
ArrayXd data(N, 1);

dpT dp;
pvT pv;

using namespace Eigen;
using namespace stan::math;

double tol = 1e-4;

int main() {
  data << 109.076, 136.503, 144.899, 184.926, 188.476, 195.562,
    199.246, 208.46 , 231.22 , 232.63 , 239.057, 241.684,
    242.159, 249.891, 266.285, 272.672, 285.217, 285.67 ,
    288.796, 296.976, 301.101, 303.024, 305.115, 305.827,
    306.939, 310.428, 318.   , 319.457, 322.249, 323.464;

  buildBasis(P, X, Y, Z, density, dp, pv);

  std::cout.precision(17);

    std::cout << "y <- c(";
  for(int i = 0; i < N; i++) {
    std::cout << data(i);

    if(i < N - 1)
      std::cout << ", ";
  }
  std::cout << ")" << std::endl;

  std::cout << "N <- " << N << std::endl; 
  std::cout << "L <- " << pv.rows() << std::endl; 

  std::cout << "dp <- c(";
  for(int i = 0; i < dp.size(); i++) {
    std::cout << dp(i);
    
    if(i < dp.size() - 1)
      std::cout << ", ";
  }
  std::cout << ")" << std::endl;

  std::cout << "pv <- structure( c(";
  for(int j = 0; j < pv.size(); j++) {
    std::cout << pv.data()[j];
    
    if(!(j == pv.size() - 1))
      std::cout << ", ";
  }
  std::cout << "), .Dim = c(" << pv.rows() << ", " << pv.cols() << ") )" << std::endl;
}

