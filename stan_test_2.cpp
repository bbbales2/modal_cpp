#include <stan/math.hpp>
#include <iostream>
#include <cmath>

#include "util.hpp"
#include "polybasis.hpp"
#include "mechanics.hpp"
#include "stan_mech.hpp"

const double X = 0.07,
  Y = 0.05,
  Z = 0.018;

const double density = 8700.0;

int P = 10;

int N = 10;

using namespace Eigen;
using namespace stan::math;

double tol = 1e-3;
double fdtol = 1e-3;

int main() {
  Matrix<double, Dynamic, 1> lookup = rus_namespace::mech_init(P, X, Y, Z, density, NULL);

  Matrix<var, Dynamic, 1> vec(3);

  Matrix<var, Dynamic, Dynamic> C(6, 6);

  var c11 = 2.5,
    a = 2.8,
    c44 = 1.4;

  var c12 = -(c44 * 2.0 / a - c11);

  C << c11, c12, c12, 0.0, 0.0, 0.0,
    c12, c11, c12, 0.0, 0.0, 0.0,
    c12, c12, c11, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, c44, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, c44, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, c44;

  double delta = 0.0001;

  auto v1 = rus_namespace::mech_rus(N, lookup, C, NULL);

  VectorXd ref(N);

  ref << 12.98034, 16.69101, 22.81702, 24.60658, 26.93973, 30.30027, 31.77975, 35.43840, 36.24037, 36.26240;

  bool failed = false;

  for(int i = 0; i < N; i++) {
    std::cout << "Computed: " << v1(i).val() << ", reference: " << ref(i);
    if(std::abs(v1(i).val() - ref(i)) / ref(i) > tol) {
      failed = true;
      std::cout << " *";
      //break;
    }
    std::cout << std::endl;
  }

  if(failed)
    std::cout << "Failed freq check" << std::endl;
  else
    std::cout << "Passed freq check" << std::endl;

  /*failed = false;

  VectorXd refc11(N);

  refc11 << 1.58096611e-02,   5.92606098e+00,   6.52767552e+00,
    1.33702240e+00,   4.99569554e+00,   1.65364703e-01,
    1.85331828e+00,   2.86127299e+00,   1.53803563e+01,
    1.26443548e+00,   3.89031218e+00,   4.88030577e+00,
    4.73043451e+00,   9.66712421e+00,   2.58175197e+00,
    3.32103660e+00,   1.18496950e+00,   3.98949689e+00,
    6.95709878e+00,   8.84615784e+00,   2.03038423e+01,
    9.48756016e+00,   2.89992732e+01,   1.07677790e+01,
    1.91150986e+01,   1.53958549e+01,   4.79186147e+00,
    2.31886719e+01,   1.34852308e+01,   5.27677204e+00;
  
  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    std::cout << c11.adj() << ", " << refc11(i) << std::endl;

    if(std::abs((c11.adj() - refc11(i)) / c11.adj()) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed c11 reference gradient check" << std::endl;
  else
    std::cout << "Passed c11 reference gradient check" << std::endl;

  failed = false;

  VectorXd refc12(N);

  refc12 << -0.503395, -48.0169, -49.5532, -7.44497,
    -95.8234, -5.91993, -17.295, -24.7766, -59.6339,
    -22.8485, -120.672, -101.35, -5.10607, -67.289,
    -144.439, -63.3056, -143.012, -91.2816, -10.6521,
    -79.2997, -66.982, -40.6559, -91.408, -46.5387,
    -84.0838, -62.4809, -154.166, -127.356, -105.808,
    -51.7811;
  
  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    std::cout << c12.adj() << ", " << refc12(i) << std::endl;

    if(std::abs((c12.adj() - refc12(i)) / c12.adj()) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed c12 reference gradient check" << std::endl;
  else
  std::cout << "Passed c12 reference gradient check" << std::endl;*/
}

