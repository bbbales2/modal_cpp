#include <stan/math.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

#include "util.hpp"
#include "polybasis.hpp"
#include "mechanics.hpp"
#include "stan_mech.hpp"

using namespace Eigen;
using namespace stan::math;

double tol = 1e-5;
double fdtol = 1e-3;

int main() {
  double delta = 0.000001;

  //Matrix<var, Dynamic, Dynamic> C = rus_namespace::mech_rotate_cu(C_, cu, NULL);

  //std::cout << C << std::endl << std::endl;

  Matrix<var, Dynamic, 1> ax(4);

  ax << 0.5, 0.3, 0.812403840463596, 0.7;
  
  auto quc = rus_namespace::ax2qu(ax);

  Matrix<var, Dynamic, 1> quref(4);

  quref << 0.9393727128473789,
    0.17144890372772567,
    0.1028693422366354,
    0.2785714956633554;

  for(int i = 0; i < 4; i++) {
    if(abs((quc(i) - quref(i)).val()) > tol) {
      std::cout << "Error in ax2qu: " << quc(i) << " " << quref(i) << std::endl;
      break;
    }
  }

  Matrix<var, Dynamic, 1> cu(3);

  cu << 0.80057441, 0.66424955, 1.02499762;

  auto hoc = rus_namespace::cu2ho(cu);

  Matrix<var, Dynamic, 1> horef(3);

  horef << 0.63906110825522655, 0.51124888660418488, 0.97338888025074799;

  for(int i = 0; i < 3; i++) {
    if(abs((hoc(i) - horef(i)).val()) > tol) {
      std::cout << "Error in cu2ho: " << hoc(i) << " " << horef(i) << std::endl;
    }
  }

  Matrix<var, Dynamic, 1> ho(3);

  ho << 0.63906110825522655, 0.51124888660418488, 0.97338888025074799;

  auto axc = rus_namespace::ho2ax(ho);

  Matrix<var, Dynamic, 1> axref(4);

  axref << 0.50251890762960527, 0.40201512610368711, 0.76541399638273189, 2.941257811266673;

  for(int i = 0; i < 4; i++) {
    if(abs((axc(i) - axref(i)).val()) > tol) {
      std::cout << "Error in ho2ax: " << axc(i) << " " << axref(i) << std::endl;
    }
  }

  cu << 0.80057441, 0.66424955, 1.02499762;

  quc = rus_namespace::cu2qu(cu, NULL);

  quref << 0.1,
    0.5,
    0.4,
    0.76157731;

  for(int i = 0; i < 4; i++) {
    if(abs((quc(i) - quref(i)).val()) > tol) {
      std::cout << "Error in cu2qu: " << quc(i) << " " << quref(i) << std::endl;
    }
  }
}

