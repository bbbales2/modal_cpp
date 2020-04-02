#ifndef polybasis_hpp_
#define polybasis_hpp_

#include <vector>
#include <cmath>
#include <Eigen/Core>

#include "util.hpp"

using namespace Eigen;
static double bp_[] = { 0.1333333333333333, 0.06666666666666666, -0.03333333333333333,
                        -0.500000000000000, 0.66666666666666666, -0.16666666666666666,
                        0.06666666666666666, 0.53333333333333333, 0.066666666666666666,
                        -0.66666666666666666, 0.000000000000000, 0.66666666666666666,
                        -0.0333333333333333, 0.066666666666666666, 0.133333333333333333,
                        0.16666666666666666, -0.66666666666666666, 0.500000000000000000,
                        -0.5000000000000000, -0.66666666666666666, 0.16666666666666666,
                        2.33333333333333333, -2.66666666666666666, 0.33333333333333333,
                        0.66666666666666666, 0.000000000000000000, -0.66666666666666666,
                        -2.66666666666666666, 5.3333333333333333, -2.66666666666666666,
                        -0.16666666666666666, 0.66666666666666666, 0.500000000000000000,
                        0.33333333333333333, -2.66666666666666666, 2.33333333333333333 };

double inner(double X, double Y, int i0, int i1, int j0, int j1) {
  if(i0 < 0 | i1 < 0 | j0 < 0 | j1 < 0 | (i0 + i1) % 2 > 0 | (j0 + j1) % 2 > 0)
    return 0.0;
  
  double ret = (2 * std::pow(0.5 * X, i0 + i1 + 1) / (i0 + i1 + 1)) *
          (2 * std::pow(0.5 * Y, j0 + j1 + 1) / (j0 + j1 + 1));

  return ret;
}

int computeBilayerSize(const int& IN, const int& JN, const int& KN) {
  int m = 0;
  for(int i = 0; i <= IN; i++) {
    for(int j = 0; j <= JN; j++) {
      for(int k = 0; k < KN; k++) {
        if((i + j) <= std::max(IN, JN)) {
          if(k < KN - 1) {
            m += 2;
          } else {
            m += 1;
          }
        }
      }
    }
  }
  
  return 3 * 3 * m * m * 36 * 2;
}

// This file contains some helper functions which build the lookup tables necessary for solving for the resonance modes of the material
// Look at Visscher 1991 (http://dx.doi.org/10.1121/1.401643) for more details

Matrix<double, 9, 9> voigt(const Matrix<double, 6, 6>& Ch) {
  Matrix<double, 9, 9> C;

  C(0 + 0 * 3, 0 + 0 * 3) = Ch(0, 0);
  C(0 + 0 * 3, 1 + 1 * 3) = Ch(0, 1);
  C(0 + 0 * 3, 2 + 2 * 3) = Ch(0, 2);
  C(0 + 0 * 3, 1 + 2 * 3) = Ch(0, 3);
  C(0 + 0 * 3, 2 + 1 * 3) = Ch(0, 3);
  C(0 + 0 * 3, 0 + 2 * 3) = Ch(0, 4);
  C(0 + 0 * 3, 2 + 0 * 3) = Ch(0, 4);
  C(0 + 0 * 3, 0 + 1 * 3) = Ch(0, 5);
  C(0 + 0 * 3, 1 + 0 * 3) = Ch(0, 5);
  C(1 + 1 * 3, 0 + 0 * 3) = Ch(1, 0);
  C(1 + 1 * 3, 1 + 1 * 3) = Ch(1, 1);
  C(1 + 1 * 3, 2 + 2 * 3) = Ch(1, 2);
  C(1 + 1 * 3, 1 + 2 * 3) = Ch(1, 3);
  C(1 + 1 * 3, 2 + 1 * 3) = Ch(1, 3);
  C(1 + 1 * 3, 0 + 2 * 3) = Ch(1, 4);
  C(1 + 1 * 3, 2 + 0 * 3) = Ch(1, 4);
  C(1 + 1 * 3, 0 + 1 * 3) = Ch(1, 5);
  C(1 + 1 * 3, 1 + 0 * 3) = Ch(1, 5);
  C(2 + 2 * 3, 0 + 0 * 3) = Ch(2, 0);
  C(2 + 2 * 3, 1 + 1 * 3) = Ch(2, 1);
  C(2 + 2 * 3, 2 + 2 * 3) = Ch(2, 2);
  C(2 + 2 * 3, 1 + 2 * 3) = Ch(2, 3);
  C(2 + 2 * 3, 2 + 1 * 3) = Ch(2, 3);
  C(2 + 2 * 3, 0 + 2 * 3) = Ch(2, 4);
  C(2 + 2 * 3, 2 + 0 * 3) = Ch(2, 4);
  C(2 + 2 * 3, 0 + 1 * 3) = Ch(2, 5);
  C(2 + 2 * 3, 1 + 0 * 3) = Ch(2, 5);
  C(1 + 2 * 3, 0 + 0 * 3) = Ch(3, 0);
  C(2 + 1 * 3, 0 + 0 * 3) = Ch(3, 0);
  C(1 + 2 * 3, 1 + 1 * 3) = Ch(3, 1);
  C(2 + 1 * 3, 1 + 1 * 3) = Ch(3, 1);
  C(1 + 2 * 3, 2 + 2 * 3) = Ch(3, 2);
  C(2 + 1 * 3, 2 + 2 * 3) = Ch(3, 2);
  C(1 + 2 * 3, 1 + 2 * 3) = Ch(3, 3);
  C(1 + 2 * 3, 2 + 1 * 3) = Ch(3, 3);
  C(2 + 1 * 3, 1 + 2 * 3) = Ch(3, 3);
  C(2 + 1 * 3, 2 + 1 * 3) = Ch(3, 3);
  C(1 + 2 * 3, 0 + 2 * 3) = Ch(3, 4);
  C(1 + 2 * 3, 2 + 0 * 3) = Ch(3, 4);
  C(2 + 1 * 3, 0 + 2 * 3) = Ch(3, 4);
  C(2 + 1 * 3, 2 + 0 * 3) = Ch(3, 4);
  C(1 + 2 * 3, 0 + 1 * 3) = Ch(3, 5);
  C(1 + 2 * 3, 1 + 0 * 3) = Ch(3, 5);
  C(2 + 1 * 3, 0 + 1 * 3) = Ch(3, 5);
  C(2 + 1 * 3, 1 + 0 * 3) = Ch(3, 5);
  C(0 + 2 * 3, 0 + 0 * 3) = Ch(4, 0);
  C(2 + 0 * 3, 0 + 0 * 3) = Ch(4, 0);
  C(0 + 2 * 3, 1 + 1 * 3) = Ch(4, 1);
  C(2 + 0 * 3, 1 + 1 * 3) = Ch(4, 1);
  C(0 + 2 * 3, 2 + 2 * 3) = Ch(4, 2);
  C(2 + 0 * 3, 2 + 2 * 3) = Ch(4, 2);
  C(0 + 2 * 3, 1 + 2 * 3) = Ch(4, 3);
  C(0 + 2 * 3, 2 + 1 * 3) = Ch(4, 3);
  C(2 + 0 * 3, 1 + 2 * 3) = Ch(4, 3);
  C(2 + 0 * 3, 2 + 1 * 3) = Ch(4, 3);
  C(0 + 2 * 3, 0 + 2 * 3) = Ch(4, 4);
  C(0 + 2 * 3, 2 + 0 * 3) = Ch(4, 4);
  C(2 + 0 * 3, 0 + 2 * 3) = Ch(4, 4);
  C(2 + 0 * 3, 2 + 0 * 3) = Ch(4, 4);
  C(0 + 2 * 3, 0 + 1 * 3) = Ch(4, 5);
  C(0 + 2 * 3, 1 + 0 * 3) = Ch(4, 5);
  C(2 + 0 * 3, 0 + 1 * 3) = Ch(4, 5);
  C(2 + 0 * 3, 1 + 0 * 3) = Ch(4, 5);
  C(0 + 1 * 3, 0 + 0 * 3) = Ch(5, 0);
  C(1 + 0 * 3, 0 + 0 * 3) = Ch(5, 0);
  C(0 + 1 * 3, 1 + 1 * 3) = Ch(5, 1);
  C(1 + 0 * 3, 1 + 1 * 3) = Ch(5, 1);
  C(0 + 1 * 3, 2 + 2 * 3) = Ch(5, 2);
  C(1 + 0 * 3, 2 + 2 * 3) = Ch(5, 2);
  C(0 + 1 * 3, 1 + 2 * 3) = Ch(5, 3);
  C(0 + 1 * 3, 2 + 1 * 3) = Ch(5, 3);
  C(1 + 0 * 3, 1 + 2 * 3) = Ch(5, 3);
  C(1 + 0 * 3, 2 + 1 * 3) = Ch(5, 3);
  C(0 + 1 * 3, 0 + 2 * 3) = Ch(5, 4);
  C(0 + 1 * 3, 2 + 0 * 3) = Ch(5, 4);
  C(1 + 0 * 3, 0 + 2 * 3) = Ch(5, 4);
  C(1 + 0 * 3, 2 + 0 * 3) = Ch(5, 4);
  C(0 + 1 * 3, 0 + 1 * 3) = Ch(5, 5);
  C(0 + 1 * 3, 1 + 0 * 3) = Ch(5, 5);
  C(1 + 0 * 3, 0 + 1 * 3) = Ch(5, 5);
  C(1 + 0 * 3, 1 + 0 * 3) = Ch(5, 5);

  return C;
}

Eigen::VectorXd buildBilayerBasis(const int& IN, const int& JN, int layer_index, double X, double Y, const Eigen::VectorXd& Z, double bulk_density, double layer_density) {
  std::vector<double> densities = { bulk_density, layer_density };

  layer_index = layer_index - 1;

  Eigen::VectorXd bp(3 * 2 * 3 * 2); // 3 x 2 x 3 x 2
  auto bp_idx = [](int i, int j, int k, int l) {
    return i + 3 * j + 3 * 2 * k + 3 * 2 * 3 * l;
  };
  for(int i = 0; i < bp.size(); i++)
    bp(i) = bp_[i];

  std::vector<std::array<int, 3> > ijsums;
  for(int i = 0; i <= IN; i++) {
    for(int j = 0; j <= JN; j++) {
      if((i + j) <= std::max(IN, JN)) {
        ijsums.push_back({ i, j, i + j });
      }
    }
  }
  
  std::sort(ijsums.begin(), ijsums.end(),
            [](const std::array<int, 3>& a, const std::array<int, 3>& b) { return a[2] > b[2]; } );
  
  std::vector<std::array<int, 3> > ntoijk;
  std::vector<int> ntom;
  std::vector<int> nlength;
  std::vector<int> mton;
  int n = 0;
  int m = 0;
  
  for(int ij = 0; ij < ijsums.size(); ij++) {
    for(int k = 0; k < Z.size(); k++) {
      int i = ijsums[ij][0];
      int j = ijsums[ij][1];
      
      ntoijk.push_back({ i, j, k });
      //std::cout << i << ", " << j << ", " << k << std::endl;
      ntom.push_back(m);
      if(k < Z.size() - 1) {
	nlength.push_back(2);
      } else {
	nlength.push_back(1);
      }
      
      for(int l = 0; l < nlength.back(); l++) {
	mton.push_back(n);
      }
      
      m = m + nlength.back();
      n = n + 1;
    }
  }

  int N = ntoijk.size();
  int M = m;

  Eigen::VectorXd dinp = Eigen::VectorXd::Zero(2 * M * M * 3 * 3); // 2 x M x M x 3 x 3
  auto dinp_idx = [M](int i, int j, int k, int l, int m) {
    return i + j * 2 + k * 2 * M + l * 2 * M * M + m * 2 * M * M * 3;
  };
  Eigen::VectorXd inp = Eigen::VectorXd::Zero(2 * M * M); // 2 x M x M
  auto inp_idx = [M](int i, int j, int k) {
    return i + j * 2 + k * 2 * M;
  };

  for(int ii = 0; ii < 2; ii++) {
    for(int n0 = 0; n0 < N; n0++) {
      for(int n1 = 0; n1 < N; n1++) {
        //std::cout << "hi" << std::endl;
        int i0 = ntoijk[n0][0];
        int j0 = ntoijk[n0][1];
        int k0 = ntoijk[n0][2];
        int i1 = ntoijk[n1][0];
        int j1 = ntoijk[n1][1];
        int k1 = ntoijk[n1][2];
        
        if(ii == 0) {
          if(k0 >= layer_index) {
            continue;
          }
        } else {
          if(k0 < layer_index) {
            continue;
          }
        }
        
        if(k0 != k1) {
          continue;
        }
        
        if(k0 == Z.size() - 1) {
          continue;
        }
        
        double dz = (Z[k0 + 1] - Z[k0]);
        //r0 = ntom[[n0]]:(ntom[[n0]] + nlength[[n0]]);
        //r1 = ntom[[n1]]:(ntom[[n1]] + nlength[[n1]]);
          
        for(int m0 = 0; m0 <= nlength[n0]; m0++) {
          for(int m1 = 0; m1 <= nlength[n1]; m1++) {
            int r0 = ntom[n0] + m0;
            int r1 = ntom[n1] + m1;
            
            //std::cout << "n0, n1: " << n0 << ", " << n1 << "; nlength[n0]: " << nlength[n0] << "; r0, r1: " << r0 << ", " << r1 << std::endl;

            double tmp = i0 * i1 * inner(X, Y, i0 - 1, i1 - 1, j0, j1); //# f0f1
            dinp(dinp_idx(ii, r0, r1, 0, 0)) = dinp(dinp_idx(ii, r0, r1, 0, 0)) + tmp * bp(bp_idx(m0, 0, m1, 0)) * dz;
            
            tmp = i0 * j1 * inner(X, Y, i0 - 1, i1, j0, j1 - 1); //# f0f1
            dinp(dinp_idx(ii, r0, r1, 0, 1)) = dinp(dinp_idx(ii, r0, r1, 0, 1)) + tmp * bp(bp_idx(m0, 0, m1, 0)) * dz;
            
            tmp = i0 * inner(X, Y, i0 - 1, i1, j0, j1); //# f0df1
            dinp(dinp_idx(ii, r0, r1, 0, 2)) = dinp(dinp_idx(ii, r0, r1, 0, 2)) + tmp * bp(bp_idx(m0, 0, m1, 1));
          
            tmp = j0 * i1 * inner(X, Y, i0, i1 - 1, j0 - 1, j1); //# f0f1
            dinp(dinp_idx(ii, r0, r1, 1, 0)) = dinp(dinp_idx(ii, r0, r1, 1, 0)) + tmp * bp(bp_idx(m0, 0, m1, 0)) * dz;
            
            tmp = j0 * j1 * inner(X, Y, i0, i1, j0 - 1, j1 - 1); //# f0f1
            dinp(dinp_idx(ii, r0, r1, 1, 1)) = dinp(dinp_idx(ii, r0, r1, 1, 1)) + tmp * bp(bp_idx(m0, 0, m1, 0)) * dz;
            
            tmp = j0 * inner(X, Y, i0, i1, j0 - 1, j1); //# f0df1
            dinp(dinp_idx(ii, r0, r1, 1, 2)) = dinp(dinp_idx(ii, r0, r1, 1, 2)) + tmp * bp(bp_idx(m0, 0, m1, 1));
          
            tmp = i1 * inner(X, Y, i0, i1 - 1, j0, j1); //# df0f1
            dinp(dinp_idx(ii, r0, r1, 2, 0)) = dinp(dinp_idx(ii, r0, r1, 2, 0)) + tmp * bp(bp_idx(m0, 1, m1, 0));
          
            tmp = j1 * inner(X, Y, i0, i1, j0, j1 - 1); //# df0f1
            dinp(dinp_idx(ii, r0, r1, 2, 1)) = dinp(dinp_idx(ii, r0, r1, 2, 1)) + tmp * bp(bp_idx(m0, 1, m1, 0));
          
            tmp = inner(X, Y, i0, i1, j0, j1); //# df0df1
            dinp(dinp_idx(ii, r0, r1, 2, 2)) = dinp(dinp_idx(ii, r0, r1, 2, 2)) + tmp * bp(bp_idx(m0, 1, m1, 1)) / dz;
            
            tmp = inner(X, Y, i0, i1, j0, j1);
            //std::cout << "(" << m0 << " " << m1 << "): " << bp(bp_idx(m0, 0, m1, 0) << std::endl;
            inp(inp_idx(ii, r0, r1)) = inp(inp_idx(ii, r0, r1)) + tmp * bp(bp_idx(m0, 0, m1, 0)) * dz;
            //std::cout << "(" << ii << " " << r0 << " " << r1 << "): " << tmp << ", " << bp(bp_idx(m0, 0, m1, 0)) << ", " << dz << ", " << inp(ii, r0, r1) << std::endl;
          }
        }
      }
    }
  }

  Eigen::VectorXd dKdcij = Eigen::VectorXd::Zero(3 * M * 3 * M * 6 * 6 * 2); // 3 * M x 3 * M x 6 x 6 x 2
  auto dKdcij_idx = [M](int i, int j, int k, int l, int m) {
    return i + j * 3 * M + k * 3 * M * 3 * M + l * 3 * M * 3 * M * 6 + m * 3 * M * 3 * M * 6 * 6;
  };
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(3 * M, 3 * M);
  
  for(int ii = 0; ii < 2; ii++) {
    for(int p = 0; p < 6; p++) {
      for(int q = 0; q < 6; q++) {
        Eigen::MatrixXd cm = Eigen::MatrixXd::Zero(6, 6);
        cm(p, q) = 1.0;
        Eigen::Matrix<double, 9, 9> c = voigt(cm);
        for(int m0 = 0; m0 < M; m0++) {
          for(int m1 = 0; m1 < M; m1++) {
            for(int i = 0; i < 3; i++) {
              for(int k = 0; k < 3; k++) {
                double total = 0.0;
                for(int j = 0; j < 3; j++) {
                  for(int l = 0; l < 3; l++) {
                    total += c(i + 3 * j, k + 3 * l) * dinp(dinp_idx(ii, m0, m1, j, l));
                  }
                }
    
                dKdcij(dKdcij_idx(3 * m0 + i, 3 * m1 + k, p, q, ii)) = dKdcij(dKdcij_idx(3 * m0 + i, 3 * m1 + k, p, q, ii)) + total;
              }
            }
          }
        }
      }
    }

    for(int m0 = 0; m0 < M; m0++) {
      for(int m1 = 0; m1 < M; m1++) {
        for(int i = 0; i < 3; i++) {
          W(3 * m0 + i, 3 * m1 + i) = W(3 * m0 + i, 3 * m1 + i) + densities[ii] * inp(inp_idx(ii, m0, m1));
        }
      }
    }
  }

  //Eigen::LLT<Eigen::MatrixXd> Wc = W.llt();
  Eigen::LDLT<Eigen::MatrixXd> Wc = W.ldlt();

  Eigen::VectorXd dKhatdcij = Eigen::VectorXd::Zero(3 * M * 3 * M * 36 * 2); // 3 * M x 3 * M x 36 x 2
  for(int ii = 0; ii < 2; ii++) {
    int ij = 0;
    for(int p = 0; p < 6; p++) {
      for(int q = 0; q < 6; q++) {
        Eigen::Map<Eigen::MatrixXd> dKtmp(&dKdcij(dKdcij_idx(0, 0, p, q, ii)), 3 * M, 3 * M);
        Eigen::Map<Eigen::MatrixXd> dKhattmp(&dKhatdcij(3 * M * 3 * M * (p + q * 6 + 36 * ii)), 3 * M, 3 * M);

	Eigen::MatrixXd K1 = (Wc.transpositionsP() * dKtmp.transpose()).transpose();
	Eigen::MatrixXd K2 = Wc.matrixL().solve(dKtmp.transpose()).transpose();
	Eigen::MatrixXd K3 = K2 * Wc.vectorD().array().sqrt().inverse().matrix().asDiagonal();

	Eigen::MatrixXd K4 = Wc.transpositionsP() * K3;
	Eigen::MatrixXd K5 = Wc.matrixL().solve(K4);
	dKhattmp = Wc.vectorD().array().sqrt().inverse().matrix().asDiagonal() * K2;
	//dKhattmp = Wc.matrixL().solve(Wc.matrixL().solve(dKtmp.transpose()).transpose());
        ij += 1;
      }
    }
  }

  /*Eigen::LDLT<Eigen::MatrixXd> Wc = W.ldlt();

  Eigen::VectorXd dKhatdcij = Eigen::VectorXd::Zero(3 * M * 3 * M * 36 * 2); // 3 * M x 3 * M x 36 x 2
  for(int ii = 0; ii < 2; ii++) {
    int ij = 0;
    for(int p = 0; p < 6; p++) {
      for(int q = 0; q < 6; q++) {
        Eigen::Map<Eigen::MatrixXd> dKtmp(&dKdcij(dKdcij_idx(0, 0, p, q, ii)), 3 * M, 3 * M);
        Eigen::Map<Eigen::MatrixXd> dKhattmp(&dKhatdcij(3 * M * 3 * M * (p + q * 6 + 36 * ii)), 3 * M, 3 * M);

	Eigen::MatrixXd PdKtmpP = Wc.transpositionsP() * (Wc.transpositionsP() * dKtmp.transpose()).transpose();
	Eigen::MatrixXd LPdKtmpPL = Wc.matrixL().solve(Wc.matrixL().solve(PdKtmpP.transpose()).transpose());
	Eigen::VectorXd D_inv = Wc.vectorD().array().inverse().sqrt();
	
	dKhattmp = D_inv.asDiagonal() * LPdKtmpPL * D_inv.asDiagonal();
        ij += 1;
      }
    }
    }*/

  return dKhatdcij;
}

MatrixXd buildK(int P, const Matrix<double, 6, 6>& Ch, const Matrix<double, Dynamic, 1>& dp) {
  int L = (P + 1) * (P + 2) * (P + 3) / 6;

  MatrixXd K = MatrixXd::Zero(L * 3, L * 3);

  Matrix<double, 9, 9> C = voigt(Ch);

  //pragma omp parallel for
  for(int n = 0; n < L; n++)
    for(int m = 0; m < L; m++) {
      Map< const Matrix<double, 3, 3> > dpm(&dp.data()[n * 3 * 3 + m * L * 3 * 3]);
      for(int i = 0; i < 3; i++)
        for(int k = 0; k < 3; k++) {
          double total = 0.0;
              
            for(int j = 0; j < 3; j++)
              for(int l = 0; l < 3; l++)
                total = total + C(i + 3 * j, k + 3 * l) * dpm(j, l);

          K(n * 3 + i, m * 3 + k) = total;
        }
    }

  return K;
}

MatrixXd buildM(int P, const Matrix<double, Dynamic, Dynamic>& pv) {
  int L = (P + 1) * (P + 2) * (P + 3) / 6;

  MatrixXd M = MatrixXd::Zero(L * 3, L * 3);

  for(int n = 0; n < L; n++) {
    for(int m = 0; m < L; m++) {
      M(n * 3 + 0, m * 3 + 0) = pv(n, m);
      M(n * 3 + 1, m * 3 + 1) = pv(n, m);
      M(n * 3 + 2, m * 3 + 2) = pv(n, m);
    }
  }

  return M;
}

double polyint(int n, int m, int l)
{
  double xtmp, ytmp;

  if(n < 0 | m < 0 | l < 0 | n % 2 > 0 | m % 2 > 0 | l % 2 > 0)
    return 0.0;
  
  xtmp = 2.0 * pow(0.5, n + 1);
  ytmp = 2.0 * xtmp * pow(0.5, m + 1);

  return 2.0 * pow(0.5, l + 1) * ytmp / ((n + 1) * (m + 1) * (l + 1));
}

void buildBasis(int P, double X, double Y, double Z, double density, Matrix<double, Dynamic, 1>& lookup) {
  std::vector<int> ns, ms, ls;

  for(int i = 0; i < P + 1; i++)
    for(int j = 0; j < P + 1; j++)
      for(int k = 0; k < P + 1; k++)
        if(i + j + k <= P) {
          ns.push_back(i);
          ms.push_back(j);
          ls.push_back(k);
        }

  int L = ns.size();

  if(L != (P + 1) * (P + 2) * (P + 3) / 6)
    throw std::logic_error("This should never happen. Make sure P is even. P = 10 or P = 12 should work");

  lookup.resize(L * L * 3 * 3 * 21);

  Matrix<double, Dynamic, 1> dp(L * L * 3 * 3);
  Matrix<double, Dynamic, Dynamic> pv(L, L);

  std::vector<double> Xs(2 * P + 3, 0.0),
    Ys(2 * P + 3, 0.0),
    Zs(2 * P + 3, 0.0);

  for(int i = -1; i < 2 * P + 2; i++) {
    Xs[i + 1] = pow(X, i);
    Ys[i + 1] = pow(Y, i);
    Zs[i + 1] = pow(Z, i);
  }

  //pragma omp parallel for
  for(int i = 0; i < ns.size(); i++) {
    for(int j = 0; j < ns.size(); j++) {
      int n0 = ns[i],
        m0 = ms[i],
        l0 = ls[i],
        n1 = ns[j],
        m1 = ms[j],
        l1 = ls[j];

      Map< Matrix<double, 3, 3> > dpm(&dp.data()[i * 3 * 3 + j * ns.size() * 3 * 3]);

      dpm(0, 0) = Xs[n1 + n0] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 2, m1 + m0, l1 + l0) * n0 * n1;
      dpm(0, 1) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * n0 * m1;
      dpm(0, 2) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * n0 * l1;

      dpm(1, 0) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * m0 * n1;
      dpm(1, 1) = Xs[n1 + n0 + 2] * Ys[m1 + m0] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0 - 2, l1 + l0) * m0 * m1;
      dpm(1, 2) = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * m0 * l1;

      dpm(2, 0) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * l0 * n1;
      dpm(2, 1) = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * l0 * m1;
      dpm(2, 2) = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0 ] * polyint(n1 + n0, m1 + m0, l1 + l0 - 2) * l0 * l1;

      pv(i, j) = density * Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0, l1 + l0);
    }
  }

  LLT< MatrixXd > M = buildM(P, pv).llt();

  int ij = 0;
  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < i + 1; j++) {
      Map< Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[ij * L * L * 3 * 3], 3 * L, 3 * L);

      Matrix<double, 6, 6> dCdcij = Matrix<double, 6, 6>::Zero();
        
      dCdcij(i, j) = 1.0;
      dCdcij(j, i) = 1.0;
        
      MatrixXd dKtmp = buildK(P, dCdcij, dp);

      dKdcij = M.matrixL().solve(M.matrixL().solve(dKtmp.transpose()).transpose());
      
      ij++;
    }
  }
}

Matrix<double, 6, 6> unvoigt(const Matrix<double, 9, 9>& Ch) {
  Matrix<double, 6, 6> C;

  C(0, 0) = Ch(0 + 0 * 3, 0 + 0 * 3);
  C(0, 1) = Ch(0 + 0 * 3, 1 + 1 * 3);
  C(0, 2) = Ch(0 + 0 * 3, 2 + 2 * 3);
  C(0, 3) = Ch(0 + 0 * 3, 1 + 2 * 3);
  C(0, 3) = Ch(0 + 0 * 3, 2 + 1 * 3);
  C(0, 4) = Ch(0 + 0 * 3, 0 + 2 * 3);
  C(0, 4) = Ch(0 + 0 * 3, 2 + 0 * 3);
  C(0, 5) = Ch(0 + 0 * 3, 0 + 1 * 3);
  C(0, 5) = Ch(0 + 0 * 3, 1 + 0 * 3);
  C(1, 0) = Ch(1 + 1 * 3, 0 + 0 * 3);
  C(1, 1) = Ch(1 + 1 * 3, 1 + 1 * 3);
  C(1, 2) = Ch(1 + 1 * 3, 2 + 2 * 3);
  C(1, 3) = Ch(1 + 1 * 3, 1 + 2 * 3);
  C(1, 3) = Ch(1 + 1 * 3, 2 + 1 * 3);
  C(1, 4) = Ch(1 + 1 * 3, 0 + 2 * 3);
  C(1, 4) = Ch(1 + 1 * 3, 2 + 0 * 3);
  C(1, 5) = Ch(1 + 1 * 3, 0 + 1 * 3);
  C(1, 5) = Ch(1 + 1 * 3, 1 + 0 * 3);
  C(2, 0) = Ch(2 + 2 * 3, 0 + 0 * 3);
  C(2, 1) = Ch(2 + 2 * 3, 1 + 1 * 3);
  C(2, 2) = Ch(2 + 2 * 3, 2 + 2 * 3);
  C(2, 3) = Ch(2 + 2 * 3, 1 + 2 * 3);
  C(2, 3) = Ch(2 + 2 * 3, 2 + 1 * 3);
  C(2, 4) = Ch(2 + 2 * 3, 0 + 2 * 3);
  C(2, 4) = Ch(2 + 2 * 3, 2 + 0 * 3);
  C(2, 5) = Ch(2 + 2 * 3, 0 + 1 * 3);
  C(2, 5) = Ch(2 + 2 * 3, 1 + 0 * 3);
  C(3, 0) = Ch(1 + 2 * 3, 0 + 0 * 3);
  C(3, 0) = Ch(2 + 1 * 3, 0 + 0 * 3);
  C(3, 1) = Ch(1 + 2 * 3, 1 + 1 * 3);
  C(3, 1) = Ch(2 + 1 * 3, 1 + 1 * 3);
  C(3, 2) = Ch(1 + 2 * 3, 2 + 2 * 3);
  C(3, 2) = Ch(2 + 1 * 3, 2 + 2 * 3);
  C(3, 3) = Ch(1 + 2 * 3, 1 + 2 * 3);
  C(3, 3) = Ch(1 + 2 * 3, 2 + 1 * 3);
  C(3, 3) = Ch(2 + 1 * 3, 1 + 2 * 3);
  C(3, 3) = Ch(2 + 1 * 3, 2 + 1 * 3);
  C(3, 4) = Ch(1 + 2 * 3, 0 + 2 * 3);
  C(3, 4) = Ch(1 + 2 * 3, 2 + 0 * 3);
  C(3, 4) = Ch(2 + 1 * 3, 0 + 2 * 3);
  C(3, 4) = Ch(2 + 1 * 3, 2 + 0 * 3);
  C(3, 5) = Ch(1 + 2 * 3, 0 + 1 * 3);
  C(3, 5) = Ch(1 + 2 * 3, 1 + 0 * 3);
  C(3, 5) = Ch(2 + 1 * 3, 0 + 1 * 3);
  C(3, 5) = Ch(2 + 1 * 3, 1 + 0 * 3);
  C(4, 0) = Ch(0 + 2 * 3, 0 + 0 * 3);
  C(4, 0) = Ch(2 + 0 * 3, 0 + 0 * 3);
  C(4, 1) = Ch(0 + 2 * 3, 1 + 1 * 3);
  C(4, 1) = Ch(2 + 0 * 3, 1 + 1 * 3);
  C(4, 2) = Ch(0 + 2 * 3, 2 + 2 * 3);
  C(4, 2) = Ch(2 + 0 * 3, 2 + 2 * 3);
  C(4, 3) = Ch(0 + 2 * 3, 1 + 2 * 3);
  C(4, 3) = Ch(0 + 2 * 3, 2 + 1 * 3);
  C(4, 3) = Ch(2 + 0 * 3, 1 + 2 * 3);
  C(4, 3) = Ch(2 + 0 * 3, 2 + 1 * 3);
  C(4, 4) = Ch(0 + 2 * 3, 0 + 2 * 3);
  C(4, 4) = Ch(0 + 2 * 3, 2 + 0 * 3);
  C(4, 4) = Ch(2 + 0 * 3, 0 + 2 * 3);
  C(4, 4) = Ch(2 + 0 * 3, 2 + 0 * 3);
  C(4, 5) = Ch(0 + 2 * 3, 0 + 1 * 3);
  C(4, 5) = Ch(0 + 2 * 3, 1 + 0 * 3);
  C(4, 5) = Ch(2 + 0 * 3, 0 + 1 * 3);
  C(4, 5) = Ch(2 + 0 * 3, 1 + 0 * 3);
  C(5, 0) = Ch(0 + 1 * 3, 0 + 0 * 3);
  C(5, 0) = Ch(1 + 0 * 3, 0 + 0 * 3);
  C(5, 1) = Ch(0 + 1 * 3, 1 + 1 * 3);
  C(5, 1) = Ch(1 + 0 * 3, 1 + 1 * 3);
  C(5, 2) = Ch(0 + 1 * 3, 2 + 2 * 3);
  C(5, 2) = Ch(1 + 0 * 3, 2 + 2 * 3);
  C(5, 3) = Ch(0 + 1 * 3, 1 + 2 * 3);
  C(5, 3) = Ch(0 + 1 * 3, 2 + 1 * 3);
  C(5, 3) = Ch(1 + 0 * 3, 1 + 2 * 3);
  C(5, 3) = Ch(1 + 0 * 3, 2 + 1 * 3);
  C(5, 4) = Ch(0 + 1 * 3, 0 + 2 * 3);
  C(5, 4) = Ch(0 + 1 * 3, 2 + 0 * 3);
  C(5, 4) = Ch(1 + 0 * 3, 0 + 2 * 3);
  C(5, 4) = Ch(1 + 0 * 3, 2 + 0 * 3);
  C(5, 5) = Ch(0 + 1 * 3, 0 + 1 * 3);
  C(5, 5) = Ch(0 + 1 * 3, 1 + 0 * 3);
  C(5, 5) = Ch(1 + 0 * 3, 0 + 1 * 3);
  C(5, 5) = Ch(1 + 0 * 3, 1 + 0 * 3);

  return C;
}

Matrix<double, 6, 6> rotate(const Matrix<double, 6, 6>& Ch, const Matrix<double, 3, 3>& Q) {
  Matrix<double, 9, 9> Cv = voigt(Ch);
  Matrix<double, 9, 9> Cr = Matrix<double, 9, 9>::Zero();

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 3; k++)
        for(int l = 0; l < 3; l++)
          for(int p = 0; p < 3; p++)
            for(int q = 0; q < 3; q++)
              for(int r = 0; r < 3; r++)
                for(int s = 0; s < 3; s++)
                  Cr(i + 3 * j, k + 3 * l) += Q(i, p) * Q(j, q) * Cv(p + 3 * q, r + 3 * s) * Q(k, r) * Q(l, s);
  
  return unvoigt(Cr);
}

Matrix<double, 6, 6> drotate(const Matrix<double, 6, 6>& Ch, const Matrix<double, 3, 3>& Q, const Matrix<double, 3, 3>& dQ) {
  Matrix<double, 9, 9> Cv = voigt(Ch);
  Matrix<double, 9, 9> Cr = Matrix<double, 9, 9>::Zero();

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 3; k++)
        for(int l = 0; l < 3; l++)
          for(int p = 0; p < 3; p++)
            for(int q = 0; q < 3; q++)
              for(int r = 0; r < 3; r++)
                for(int s = 0; s < 3; s++) {
                  Cr(i + 3 * j, k + 3 * l) += dQ(i, p) * Q(j, q) * Cv(p + 3 * q, r + 3 * s) * Q(k, r) * Q(l, s) +
                    Q(i, p) * dQ(j, q) * Cv(p + 3 * q, r + 3 * s) * Q(k, r) * Q(l, s) +
                    Q(i, p) * Q(j, q) * Cv(p + 3 * q, r + 3 * s) * dQ(k, r) * Q(l, s) +
                    Q(i, p) * Q(j, q) * Cv(p + 3 * q, r + 3 * s) * Q(k, r) * dQ(l, s);
                }
  
  return unvoigt(Cr);
}

void buildRotate(const Matrix<double, 6, 6>& Ch, double w, double x, double y, double z,
                 Matrix<double, 6, 6>& Cr,
                 Matrix<double, 6, 6>& dCrdw,
                 Matrix<double, 6, 6>& dCrdx,
                 Matrix<double, 6, 6>& dCrdy,
                 Matrix<double, 6, 6>& dCrdz,
                 Matrix<double, 3, 3>& Q) {
  // Code adapted from Will Lenthe
  Matrix<double, 3, 3> dQdw, dQdx, dQdy, dQdz;
  
  Q << w * w - (y * y + z * z) + x * x, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
    2.0 * (y * x + w * z), w * w - (x * x + z * z) + y * y, 2.0 * (y * z - w * x),
    2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w * w - (x * x + y * y) + z * z;

  dQdw << 2 * w, -2.0 * z, 2.0 * y,
    2.0 * z, 2 * w, -2.0 * x,
    -2.0 * y, 2.0 * x, 2 * w;
  
  dQdx << 2 * x, 2.0 * y, 2.0 * z,
    2.0 * y, -2.0 * x, -2.0 * w,
    2.0 * z, 2.0 * w, -2.0 * x;

  dQdy << -2 * y, 2 * x, 2 * w,
    2 * x, 2 * y, 2 * z,
    -2 * w, 2 * z, -2 * y;

  dQdz << -2 * z, -2 * w, 2 * x,
    2 * w, -2 * z, 2 * y,
    2 * x, 2 * y, 2 * z;

  Cr = rotate(Ch, Q);
  dCrdw = drotate(Ch, Q, dQdw);
  dCrdx = drotate(Ch, Q, dQdx);
  dCrdy = drotate(Ch, Q, dQdy);
  dCrdz = drotate(Ch, Q, dQdz);
}

void buildKM(int P, const Matrix<double, 6, 6>& Ch, const Matrix<double, Dynamic, 1>& lookup, MatrixXd& K_, MatrixXd& M_) {
  int L = (P + 1) * (P + 2) * (P + 3) / 6;

  K_.resize(L * 3, L * 3);
  K_.setZero();
  
  int ij = 0;
  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < i + 1; j++) {
      Map< const Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[L * L * 3 * 3 + L * L + ij * L * L * 3 * 3], 3 * L, 3 * L);

      K_ += dKdcij * Ch(i, j);

      ij++;
    }
  }

  Map< const Matrix<double, Dynamic, Dynamic> > M(&lookup.data()[L * L * 3 * 3 + L * L + 21 * L * L * 3 * 3], L * 3, L * 3);

  M_ = M;
}

#endif

