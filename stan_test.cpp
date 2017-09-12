#include <stan/math.hpp>
#include <iostream>
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

using namespace Eigen;
using namespace stan::math;

double tol = 1e-5;
double fdtol = 1e-3;

int main() {
  data << 109.076, 136.503, 144.899, 184.926, 188.476, 195.562,
    199.246, 208.46 , 231.22 , 232.63 , 239.057, 241.684,
    242.159, 249.891, 266.285, 272.672, 285.217, 285.67 ,
    288.796, 296.976, 301.101, 303.024, 305.115, 305.827,
    306.939, 310.428, 318.   , 319.457, 322.249, 323.464;

  Matrix<double, Dynamic, 1> lookup = rus_namespace::mech_init(P, X, Y, Z, density, NULL);

  Matrix<var, Dynamic, 1> vec(3);

  Matrix<var, Dynamic, Dynamic> C(6, 6);

  var c11 = 1.6,
    a = 1.0,
    c44 = 0.446;

  var c12 = -(c44 * 2.0 / a - c11);

  C << c11, c12, c12, 0.0, 0.0, 0.0,
    c12, c11, c12, 0.0, 0.0, 0.0,
    c12, c12, c11, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, c44, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, c44, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, c44;

  double delta = 0.00001;

  auto v1 = rus_namespace::mech_rus(P, N, lookup, C, NULL);

  VectorXd ref(N);

  ref << 108.47023141, 135.00692667, 143.37988477, 183.83960174,
    186.95853441, 194.85994545, 198.32368438, 207.20001131,
    229.56350053, 231.65836965, 237.96801498, 241.04923862,
    241.28543212, 248.26841215, 266.09547338, 271.28146467,
    283.99770118, 284.43249363, 287.6396657, 295.07029835,
    298.86837277, 301.45603222, 301.74432735, 303.1395348,
    303.78519859, 307.95540376, 316.03191991, 316.92235217,
    319.99504521, 321.69228994;

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

  int ij = 0;
  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < i + 1; j++) {
      Matrix<var, Dynamic, Dynamic> Ct = C;

      Ct(i, j) = C(i, j) + delta;
      Ct(j, i) = Ct(i, j);

      //std::cout << Ct << std::endl << "--" << std::endl;
      //std::cout << C << std::endl << "**" << std::endl;

      auto v2 = rus_namespace::mech_rus(P, N, lookup, Ct, NULL);
      
      failed = false;
      
      for(int n = 0; n < N; n++) {
        set_zero_all_adjoints();
        
        v2(n).grad();

        var rel = abs((Ct(i, j).adj() - (v2(n) - v1(n)) / delta) / Ct(i, j).adj());
        
        if(rel > fdtol) {
          failed = true;

          std::cout << Ct(i, j).adj() << " " << (v2(n) - v1(n)) / delta << std::endl;
          std::cout << rel << " " << fdtol << std::endl;
          
          break;
        }
      }

      if(failed)
        std::cout << "Failed fd gradient check (" << i << ", " << j << ")" << std::endl;
      else
        std::cout << "Passed fd gradient check (" << i << ", " << j << ")" << std::endl;

      //std::cout << "========" << std::endl;

      ij++;
    }
  }

  failed = false;

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

