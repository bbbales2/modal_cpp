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

  Matrix<var, Dynamic, Dynamic> C_(6, 6);

  var c11 = 1.6,
    a = 1.0,
    c44 = 0.446;

  var c12 = -(c44 * 2.0 / a - c11);

  C_ << c11, c12, c12, 0.0, 0.0, 0.0,
    c12, c11, c12, 0.0, 0.0, 0.0,
    c12, c12, c11, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, c44, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, c44, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, c44;

  Matrix<var, Dynamic, 1> q(4);

  q(0) = 0.1;
  q(1) = 0.5;
  q(2) = 0.4;
  q(3) = sqrt(1.0 - 0.1 * 0.1 - 0.5 * 0.5 - 0.4 * 0.4);

  double delta = 0.000001;

  Matrix<var, Dynamic, Dynamic> C = rus_namespace::mech_rotate(C_, q, NULL);

  //std::cout << C << std::endl << std::endl;
  
  auto v1 = rus_namespace::mech_rus(N, lookup, C, NULL);

  VectorXd ref(N);

  ref << 108.47023141,  135.00692667,  143.37988477,  183.83960174,
    186.95853441,  194.85994545,  198.32368438,  207.20001131,
    229.56350053,  231.65836965,  237.96801498,  241.04923862,
    241.28543212,  248.26841215,  266.09547338,  271.28146467,
    283.99770118,  284.43249363,  287.6396657 ,  295.07029835,
    298.86837277,  301.45603222,  301.74432735,  303.1395348 ,
    303.78519859,  307.95540376,  316.03191991,  316.92235217,
    319.99504521,  321.69228994;

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

  /*int ij = 0;
  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < i + 1; j++) {
      Matrix<var, Dynamic, Dynamic> Ct = C;

      var tmp = C(i, j) + delta;
      
      Ct(i, j) = tmp;
      Ct(j, i) = tmp;

      //std::cout << Ct << std::endl << "--" << std::endl;
      //std::cout << C << std::endl << "**" << std::endl;

      auto v2 = test_model_namespace::mech(N, dp, pv, Ct, NULL);
      
      failed = false;
      
      for(int n = 0; n < N; n++) {
        set_zero_all_adjoints();
        
        v2(n).grad();

        var rel = abs((tmp.adj() - (v2(n) - v1(n)) / delta) / tmp.adj());
        
        if(rel > fdtol) {
          failed = true;

          //std::cout << tmp.adj() << " " << (v2(n) - v1(n)) / delta << std::endl;
          //std::cout << rel << " " << fdtol << std::endl;
          
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
    }*/

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

  failed = false;

  VectorXd refw(N);

  refw << 43.38809256,   54.00277067,   57.35195391,   73.5358407 ,
    74.78341376,   77.94397818,   79.32947375,   82.88000452,
    91.82540021,   92.66334786,   95.18720599,   96.41969545,
    96.51417285,   99.30736486,  106.43818935,  108.51258587,
    113.59908047,  113.77299745,  115.05586628,  118.02811934,
    119.54734911,  120.58241289,  120.69773094,  121.25581392,
    121.51407944,  123.1821615 ,  126.41276796,  126.76894087,
    127.99801808,  128.67691598;

  var& w = q(0);
  
  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    if(abs((w.adj() - refw(i)) / w.adj()) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed w reference gradient check" << std::endl;
  else
    std::cout << "Passed w reference gradient check" << std::endl;

  failed = false;

  VectorXd refx(N);

  refx << 216.94046282,  270.01385333,  286.75976954,  367.67920349,
    373.91706882,  389.71989089,  396.64736876,  414.40002262,
    459.12700105,  463.31673929,  475.93602997,  482.09847723,
    482.57086424,  496.5368243 ,  532.19094676,  542.56292934,
    567.99540236,  568.86498726,  575.27933141,  590.1405967 ,
    597.73674554,  602.91206443,  603.4886547 ,  606.2790696 ,
    607.57039719,  615.91080751,  632.06383982,  633.84470434,
    639.99009042,  643.38457988;

  var& x = q(1);
  
  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    if(abs((x.adj() - refx(i)) / x.adj()) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed x reference gradient check" << std::endl;
  else
    std::cout << "Passed x reference gradient check" << std::endl;

  failed = false;

  VectorXd refy(N);

  refy << 173.55237026,  216.01108267,  229.40781563,  294.14336279,
    299.13365506,  311.77591272,  317.31789501,  331.5200181 ,
    367.30160084,  370.65339143,  380.74882398,  385.67878179,
    386.05669139,  397.22945944,  425.7527574 ,  434.05034347,
    454.39632189,  455.09198981,  460.22346513,  472.11247736,
    478.18939643,  482.32965155,  482.79092376,  485.02325568,
    486.05631775,  492.72864601,  505.65107186,  507.07576347,
    511.99207234,  514.7076639;

  var& y = q(2);
  
  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    if(abs((y.adj() - refy(i)) / y.adj()) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed y reference gradient check" << std::endl;
  else
    std::cout << "Passed y reference gradient check" << std::endl;

  failed = false;

  VectorXd refz(N);

  refz << 330.43386847,  411.27284848,  436.77946813,  560.0322779 ,
    569.53351132,  593.60365278,  604.1552727 ,  631.19530947,
    699.32141336,  705.70303252,  724.92416343,  734.31052346,
    735.03004191,  756.30235851,  810.6090999 ,  826.4072331 ,
    865.14482191,  866.46933417,  876.2393721 ,  898.87537701,
    910.44548621,  918.3282971 ,  919.20653323,  923.45676659,
    925.42365816,  938.12739269,  962.7309585 ,  965.44349052,
    974.80386373,  979.97419603;

  var& z = q(3);
  
  for(int i = 0; i < N; i++) {
    set_zero_all_adjoints();
    
    v1(i).grad();

    if(abs((z.adj() - refz(i)) / z.adj()) > tol) {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "Failed z reference gradient check" << std::endl;
  else
    std::cout << "Passed z reference gradient check" << std::endl;
}

