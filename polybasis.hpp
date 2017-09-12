#ifndef polybasis_hpp_
#define polybasis_hpp_

#include <vector>
#include <cmath>
//#include <Eigen/Core>

#include "util.hpp"

using namespace Eigen;

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

void buildK(int P, const Matrix<double, 6, 6>& Ch, const Matrix<double, Dynamic, 1>& lookup, Map< MatrixXd >& K) {
  int L = (P + 1) * (P + 2) * (P + 3) / 6;

  K.resize(L * 3, L * 3);
  K.setZero();

  Map< const Matrix<double, Dynamic, 1> > dp(&lookup.data()[0], L * L * 3 * 3);

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
}

void buildM(int P, const Matrix<double, Dynamic, 1>& lookup, Map< MatrixXd >& M) {
  int L = (P + 1) * (P + 2) * (P + 3) / 6;

  Map< const Matrix<double, Dynamic, Dynamic> > pv(&lookup.data()[L * L * 3 * 3], L, L);
  
  M.resize(L * 3, L * 3);
  M.setZero();

  for(int n = 0; n < L; n++) {
    for(int m = 0; m < L; m++) {
      M(n * 3 + 0, m * 3 + 0) = pv(n, m);
      M(n * 3 + 1, m * 3 + 1) = pv(n, m);
      M(n * 3 + 2, m * 3 + 2) = pv(n, m);
    }
  }
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

  lookup.resize(L * L * 3 * 3 + L * L + L * L * 3 * 3 * 21 + L * L * 3 * 3);

  Map< Matrix<double, Dynamic, 1> > dp(&lookup.data()[0], L * L * 3 * 3);
  Map< Matrix<double, Dynamic, Dynamic> > pv(&lookup.data()[L * L * 3 * 3], L, L);

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

  int ij = 0;
  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < i + 1; j++) {
      Map< Matrix<double, Dynamic, Dynamic> > dKdcij(&lookup.data()[L * L * 3 * 3 + L * L + ij * L * L * 3 * 3], 3 * L, 3 * L);

      Matrix<double, 6, 6> dCdcij = Matrix<double, 6, 6>::Zero();
        
      dCdcij(i, j) = 1.0;
      dCdcij(j, i) = 1.0;
        
      buildK(P, dCdcij, lookup, dKdcij);

      ij++;
    }
  }

  Map< Matrix<double, Dynamic, Dynamic> > M(&lookup.data()[L * L * 3 * 3 + L * L + 21 * L * L * 3 * 3], L * 3, L * 3);

  buildM(P, lookup, M);
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

