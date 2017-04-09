#include <stan/math.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

#include "util.hpp"
#include "polybasis.hpp"
#include "mechanics.hpp"
#include "stan_mech.hpp"

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

  Matrix<var, Dynamic, 1> vec(3);

  var c11 = 1.6,
    a = 1.0,
    c44 = 0.446,
    w = 0.1,
    x = 0.5,
    y = 0.4,
    z = std::sqrt(1 - 0.1 * 0.1 + 0.5 * 0.5 + 0.4 * 0.4);

  Matrix<var, Dynamic, 1> q(4);
  q << 0.5, 0.5, 0.5, 0.5;

  double delta = 0.00001;

  auto v1 = test_model_namespace::mechr(N, dp, pv, c11, a, c44, w, x, y, z, NULL);
  auto v2 = test_model_namespace::mechr(N, dp, pv, c11 + delta, a, c44, w, x, y, z, NULL);
  auto v3 = test_model_namespace::mechr(N, dp, pv, c11, a + delta, c44, w, x, y, z, NULL);
  auto v4 = test_model_namespace::mechr(N, dp, pv, c11, a, c44 + delta, w, x, y, z, NULL);
  auto v5 = test_model_namespace::mechr(N, dp, pv, c11, a, c44, w + delta, x, y, z, NULL);
  auto v6 = test_model_namespace::mechr(N, dp, pv, c11, a, c44, w, x + delta, y, z, NULL);
  auto v7 = test_model_namespace::mechr(N, dp, pv, c11, a, c44, w, x, y + delta, z, NULL);
  auto v8 = test_model_namespace::mechr(N, dp, pv, c11, a, c44, w, x, y, z + delta, NULL);

  VectorXd ref(N);

  ref << 359.29679453,   447.19694389,   474.9315303 ,   608.95029681,
    619.28144939,   645.4540833 ,   656.92737214,   686.32931747,
    760.40613914,   767.34518362,   788.24525284,   798.45149799,
    799.23386535,   822.3642884 ,   881.41464602,   898.59272357,
    940.71398539,   942.1541919 ,   952.77762868,   977.39085626,
    989.97159796,   998.54296111,   999.49790991,  1004.11939508,
    1006.25809182,  1020.0714794 ,  1046.82413151,  1049.77359933,
    1059.95158776,  1065.57354119;

  bool failed = false;

  for(int i = 0; i < N; i++) {
    if(abs(v1(i) - ref(i)) / ref(i) > tol) {
      failed = true;
      break;
    }
    //std::cout << v1(i) << " " << ref(i) << std::endl;
  }

  if(failed)
    std::cout << "Failed freq check" << std::endl;
  else
    std::cout << "Passed freq check" << std::endl;

  failed = false;

  for(int i = 0; i < N; i++) {
    if(i > 0)
      set_zero_all_adjoints();
    
    v1(i).grad();
    
    if(abs((c11.adj() - (v2(i) - v1(i)) / delta) / c11.adj()) > tol) {
      failed = true;
      break;
    }

    if(abs((a.adj() - (v3(i) - v1(i)) / delta) / a.adj()) > tol) {
      failed = true;
      break;
    }

    if(abs((c44.adj() - (v4(i) - v1(i)) / delta) / c44.adj()) > tol) {
      failed = true;
      break;
    }

    if(abs((w.adj() - (v5(i) - v1(i)) / delta) / w.adj()) > tol) {
      failed = true;
      break;
    }

    if(abs((x.adj() - (v6(i) - v1(i)) / delta) / x.adj()) > tol) {
      failed = true;
      break;
    }
    
    if(abs((y.adj() - (v7(i) - v1(i)) / delta) / y.adj()) > tol) {
      failed = true;
      break;
    }
    
    if(abs((z.adj() - (v8(i) - v1(i)) / delta) / z.adj()) > tol) {
      failed = true;
      break;
    }
}

  if(failed)
    std::cout << "Failed fd gradient check" << std::endl;
  else
    std::cout << "Passed fd gradient check" << std::endl;

  failed = false;

  VectorXd refc11(N);

  refc11 << 5.23679215e-02,   1.96294844e+01,   2.16222724e+01,
    4.42875299e+00,   1.65477419e+01,   5.47754041e-01,
    6.13893145e+00,   9.47768067e+00,   5.09458921e+01,
    4.18831610e+00,   1.28862701e+01,   1.61655248e+01,
    1.56690913e+01,   3.20213822e+01,   8.55179523e+00,
    1.10006016e+01,   3.92509298e+00,   1.32148095e+01,
    2.30446940e+01,   2.93020132e+01,   6.72544472e+01,
    3.14265943e+01,   9.60571925e+01,   3.56671911e+01,
    6.33168525e+01,   5.09972297e+01,   1.58725619e+01,
    7.68101568e+01,   4.46684784e+01,   1.74787797e+01;
  
  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    if(abs((c11.adj() - refc11(i)) / c11.adj()) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed c11 reference gradient check" << std::endl;
  else
  std::cout << "Passed c11 reference gradient check" << std::endl;
}

