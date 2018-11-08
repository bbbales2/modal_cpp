#include <stan/math.hpp>
#include <iostream>
#include <cmath>

#include "util.hpp"
#include "polybasis.hpp"
#include "mechanics.hpp"
#include "stan_mech.hpp"

const double X = 0.07,
  Y = 0.05,
  Z = 0.018,
  B = 0.014;

const double density = 8700.0;

int IN = 5;
int JN = 5;
int KN = 5;

int N = 10;

using namespace Eigen;
using namespace stan::math;

double tol = 1e-3;
double fdtol = 1e-3;

int main() {
  int layer_index = 5;
  Matrix<double, Dynamic, 1> Zs(KN + 1);
  Zs << 0.0000, 0.0045, 0.0090, 0.0135, 0.0140, 0.0180;
  
  Matrix<double, Dynamic, 1> lookup = rus_namespace::bilayer_init(IN, JN, layer_index, X, Y, Zs, density, density, NULL);

  Matrix<var, Dynamic, 1> vec(3);

  Matrix<var, Dynamic, Dynamic> C1(6, 6),
    C2(6, 6);

  var c12 = -1.57676;

  C1 << 5.77877, c12, 1.51418, -1.38104, 1.60109, 0.162966,
    c12, 4.6482, -1.46722, 2.83411, 0.223867, -2.81944,
    1.51418, -1.46722, 4.29902, -2.31215, -2.39193, 1.49025,
    -1.38104, 2.83411, -2.31215, 4.55989, 0.553293, -1.33558,
    1.60109, 0.223867, -2.39193, 0.553293, 4.28768, 0.638419,
    0.162966, -2.81944, 1.49025, -1.33558, 0.638419, 9.8315;

  C2 << 3.69762,-0.429256,-0.86379,-0.337691,-1.55673,0.0315084,-0.429256,3.24644,-0.265911,0.186542,0.00307126,0.105072,-0.86379,-0.265911,4.14444,-0.289584,0.0512344,-0.835652,-0.337691,0.186542,-0.289584,4.35637,0.172839,-1.02667,-1.55673,0.00307126,0.0512344,0.172839,3.96744,-0.147433,0.0315084,0.105072,-0.835652,-1.02667,-0.147433,4.00015;

  double delta = 0.0001;

  auto v1 = rus_namespace::bilayer_rus(N, lookup, C1, C2, NULL);

  VectorXd ref(N);

  ref << 17.77809, 26.06688, 32.16866, 36.00596, 36.66907, 41.56851, 44.52392, 47.79816, 50.18228, 55.42497;

  bool failed = false;

  for(int i = 0; i < N; i++) {
    std::cout << "Computed: " << v1(i).val() << ", reference: " << ref(i) << std::endl;
    if(std::abs(v1(i).val() - ref(i)) / ref(i) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed freq check" << std::endl;
  else
    std::cout << "Passed freq check" << std::endl;

  failed = false;

  VectorXd refc11(N);

  refc11 << 1.8695421, 1.8739192, 1.4910571,
    4.2128770, 1.6315171, 3.5389459,
    1.7234975, 2.2755255, 2.3960308, 0.9867144;

  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    var c11 = C1(0, 0);

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

  refc12 << 1.484821, 1.580671, 2.002227, 3.475963, 2.347846,
    3.634384, 4.379863, 2.606563, 4.107855, -3.074294;

  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    std::cout << c12.adj() << ", " << refc12(i) << std::endl;

    if(std::abs((c12.adj() - refc12(i)) / c12.adj()) > tol) {
      failed = true;
      //break;
    }
  }

  if(failed)
    std::cout << "Failed c12 reference gradient check" << std::endl;
  else
    std::cout << "Passed c12 reference gradient check" << std::endl;
}

