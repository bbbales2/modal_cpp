#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <stdio.h>
#include <iostream>

#include "magma.h"

#include "polybasis.hpp"
#include "mechanics.hpp"

const double etol = 1e-2; //10hz

int main(int argc, char **argv) {
  magma_init();
  
  double X = 0.007753,
    Y = 0.009057,
    Z = 0.013199;

  double c11 = 1.685;
  double anisotropic = 1.0;
  double c44 = 0.446;

  double density = 4401.695921;

  Eigen::Tensor<double, 4> *dp;
  Eigen::Tensor<double, 2> *pv;
  
  double tmp = omp_get_wtime();
  buildBasis(10, X, Y, Z, density, &dp, &pv);
  if(DEBUG)
    printf("buildBasis %f\n", omp_get_wtime() - tmp);

  MatrixXd *eigs, *deigs_dc11, *deigs_da, *deigs_dc44, *_;

  int nevs = 8;

  //tmp = omp_get_wtime();
  mechanics(c11, anisotropic, c44, // Params
            dp, pv, nevs, // Ref data
            &eigs, // Output
            &deigs_dc11, &deigs_da, &deigs_dc44); // Gradients

  bool passed = true;

  double ref_eigs[nevs] = { 108.47148279, 135.47169672, 143.89226719, 183.94392444,
                            187.34528646, 194.87291653, 198.46800545, 207.42295587 };

  for(int i = 0; i < nevs; i++) {
    if(std::abs((*eigs)(i, 0) - ref_eigs[i]) > etol) {
      passed = false;
    }
  }
  
  if(passed) {
    printf("eigenvalue test passed\n");
  } else {
    printf("eigenvalue test failed\n");
  }

  passed = true;

  double ref_deigs_dc11[nevs] = { 0.01370952, 5.04495188, 5.56645609, 1.12663248,
                                  4.14376517, 0.14082727, 1.55553782, 2.40418028 };

  for(int i = 0; i < nevs; i++) {
    if(std::abs((*deigs_dc11)(i, 0) - ref_deigs_dc11[i]) > etol) {
      passed = false;
    }
  }

  if(passed) {
    printf("derivative of eigenvalue test passed\n");
  } else {
    printf("derivative of eigenvalue test failed\n");
  }

  passed = true;

  MatrixXd *eigst_c11, *eigst_a, *eigst_c44;
  double delta = 0.00001;
  
  mechanics(c11 + delta, anisotropic, c44, // Params
            dp, pv, nevs, // Ref data
            &eigst_c11); // Output

  mechanics(c11, anisotropic + delta, c44, // Params
            dp, pv, nevs, // Ref data
            &eigst_a); // Output

  mechanics(c11, anisotropic, c44 + delta, // Params
            dp, pv, nevs, // Ref data
            &eigst_c44); // Output

  for(int i = 0; i < nevs; i++) {
    double approxd = ((*eigst_c11)(i, 0) - (*eigs)(i, 0)) / delta;
    //printf("c11 %f %f\n", approxd, (*deigs_dc11)(i, 0));
    if(std::abs(approxd - (*deigs_dc11)(i, 0)) > etol) {
      passed = false;
    }

    approxd = ((*eigst_a)(i, 0) - (*eigs)(i, 0)) / delta;
    //printf("a   %f %f\n", approxd, (*deigs_da)(i, 0));
    if(std::abs(approxd - (*deigs_da)(i, 0)) > etol) {
      passed = false;
    }

    approxd = ((*eigst_c44)(i, 0) - (*eigs)(i, 0)) / delta;
    //printf("c44 %f %f\n", approxd, (*deigs_dc44)(i, 0));
    if(std::abs(approxd - (*deigs_dc44)(i, 0)) > etol) {
      passed = false;
    }
  }  

  if(passed) {
    printf("gradient check passed\n");
  } else {
    printf("gradient check failed\n");
  }

  delete dp;
  delete pv;
  delete eigs;

  delete deigs_dc11;
  delete deigs_da;
  delete deigs_dc44;

  delete eigst_c11;
  delete eigst_a;
  delete eigst_c44;
  
  return 0;
}
