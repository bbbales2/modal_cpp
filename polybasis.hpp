#ifndef polybasis_hpp_
#define polybasis_hpp_

#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include "util.hpp"

double polyint(int n, int m, int l)
{
  double xtmp, ytmp;

  if(n < 0 | m < 0 | l < 0 | n % 2 > 0 | m % 2 > 0 | l % 2 > 0)
    return 0.0;
  
  xtmp = 2.0 * pow(0.5, n + 1);
  ytmp = 2.0 * xtmp * pow(0.5, m + 1);

  return 2.0 * pow(0.5, l + 1) * ytmp / ((n + 1) * (m + 1) * (l + 1));
}

void buildBasis(int N, double X, double Y, double Z, Eigen::Tensor<double, 4> **dp_, Eigen::Tensor<double, 2> **pv_) {
  std::vector<int> ns, ms, ls;

  for(int i = 0; i < N + 1; i++)
    for(int j = 0; j < N + 1; j++)
      for(int k = 0; k < N + 1; k++)
        if(i + j + k <= N) {
          ns.push_back(i);
          ms.push_back(j);
          ls.push_back(k);
        }

  Eigen::Tensor<double, 4> &dp = *new Eigen::Tensor<double, 4>(ns.size(), ns.size(), 3, 3);
  Eigen::Tensor<double, 2> &pv = *new Eigen::Tensor<double, 2>(ns.size(), ns.size());

  std::vector<double> Xs(2 * N + 3, 0.0),
    Ys(2 * N + 3, 0.0),
    Zs(2 * N + 3, 0.0);

  for(int i = -1; i < 2 * N + 2; i++) {
    Xs[i + 1] = pow(X, i);
    Ys[i + 1] = pow(Y, i);
    Zs[i + 1] = pow(Z, i);
  }

  #pragma omp parallel for
  for(int i = 0; i < ns.size(); i++) {
    for(int j = 0; j < ns.size(); j++) {
      int n0 = ns[i],
        m0 = ms[i],
        l0 = ls[i],
        n1 = ns[j],
        m1 = ms[j],
        l1 = ls[j];

      dp(i, j, 0, 0) = Xs[n1 + n0] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 2, m1 + m0, l1 + l0) * n0 * n1;
      dp(i, j, 0, 1) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * n0 * m1;
      dp(i, j, 0, 2) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * n0 * l1;

      dp(i, j, 1, 0) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * m0 * n1;
      dp(i, j, 1, 1) = Xs[n1 + n0 + 2] * Ys[m1 + m0] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0 - 2, l1 + l0) * m0 * m1;
      dp(i, j, 1, 2) = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * m0 * l1;

      dp(i, j, 2, 0) = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * l0 * n1;
      dp(i, j, 2, 1) = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * l0 * m1;
      dp(i, j, 2, 2) = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0 ] * polyint(n1 + n0, m1 + m0, l1 + l0 - 2) * l0 * l1;

      pv(i, j) = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0, l1 + l0);
    }
  }

  *dp_ = &dp;
  *pv_ = &pv;
}

void buildKM(Eigen::Tensor<double, 2> &Ch, Eigen::Tensor<double, 4> &dp, Eigen::Tensor<double, 2> &pv, double density, MatrixXd **K_, MatrixXd **M_) {
  int N = dp.dimension(0);

  Eigen::Tensor<double, 4> C(3, 3, 3, 3);
  
  C(0, 0, 0, 0) = Ch(0, 0);
  C(0, 0, 1, 1) = Ch(0, 1);
  C(0, 0, 2, 2) = Ch(0, 2);
  C(0, 0, 1, 2) = Ch(0, 3);
  C(0, 0, 2, 1) = Ch(0, 3);
  C(0, 0, 0, 2) = Ch(0, 4);
  C(0, 0, 2, 0) = Ch(0, 4);
  C(0, 0, 0, 1) = Ch(0, 5);
  C(0, 0, 1, 0) = Ch(0, 5);
  C(1, 1, 0, 0) = Ch(1, 0);
  C(1, 1, 1, 1) = Ch(1, 1);
  C(1, 1, 2, 2) = Ch(1, 2);
  C(1, 1, 1, 2) = Ch(1, 3);
  C(1, 1, 2, 1) = Ch(1, 3);
  C(1, 1, 0, 2) = Ch(1, 4);
  C(1, 1, 2, 0) = Ch(1, 4);
  C(1, 1, 0, 1) = Ch(1, 5);
  C(1, 1, 1, 0) = Ch(1, 5);
  C(2, 2, 0, 0) = Ch(2, 0);
  C(2, 2, 1, 1) = Ch(2, 1);
  C(2, 2, 2, 2) = Ch(2, 2);
  C(2, 2, 1, 2) = Ch(2, 3);
  C(2, 2, 2, 1) = Ch(2, 3);
  C(2, 2, 0, 2) = Ch(2, 4);
  C(2, 2, 2, 0) = Ch(2, 4);
  C(2, 2, 0, 1) = Ch(2, 5);
  C(2, 2, 1, 0) = Ch(2, 5);
  C(1, 2, 0, 0) = Ch(3, 0);
  C(2, 1, 0, 0) = Ch(3, 0);
  C(1, 2, 1, 1) = Ch(3, 1);
  C(2, 1, 1, 1) = Ch(3, 1);
  C(1, 2, 2, 2) = Ch(3, 2);
  C(2, 1, 2, 2) = Ch(3, 2);
  C(1, 2, 1, 2) = Ch(3, 3);
  C(1, 2, 2, 1) = Ch(3, 3);
  C(2, 1, 1, 2) = Ch(3, 3);
  C(2, 1, 2, 1) = Ch(3, 3);
  C(1, 2, 0, 2) = Ch(3, 4);
  C(1, 2, 2, 0) = Ch(3, 4);
  C(2, 1, 0, 2) = Ch(3, 4);
  C(2, 1, 2, 0) = Ch(3, 4);
  C(1, 2, 0, 1) = Ch(3, 5);
  C(1, 2, 1, 0) = Ch(3, 5);
  C(2, 1, 0, 1) = Ch(3, 5);
  C(2, 1, 1, 0) = Ch(3, 5);
  C(0, 2, 0, 0) = Ch(4, 0);
  C(2, 0, 0, 0) = Ch(4, 0);
  C(0, 2, 1, 1) = Ch(4, 1);
  C(2, 0, 1, 1) = Ch(4, 1);
  C(0, 2, 2, 2) = Ch(4, 2);
  C(2, 0, 2, 2) = Ch(4, 2);
  C(0, 2, 1, 2) = Ch(4, 3);
  C(0, 2, 2, 1) = Ch(4, 3);
  C(2, 0, 1, 2) = Ch(4, 3);
  C(2, 0, 2, 1) = Ch(4, 3);
  C(0, 2, 0, 2) = Ch(4, 4);
  C(0, 2, 2, 0) = Ch(4, 4);
  C(2, 0, 0, 2) = Ch(4, 4);
  C(2, 0, 2, 0) = Ch(4, 4);
  C(0, 2, 0, 1) = Ch(4, 5);
  C(0, 2, 1, 0) = Ch(4, 5);
  C(2, 0, 0, 1) = Ch(4, 5);
  C(2, 0, 1, 0) = Ch(4, 5);
  C(0, 1, 0, 0) = Ch(5, 0);
  C(1, 0, 0, 0) = Ch(5, 0);
  C(0, 1, 1, 1) = Ch(5, 1);
  C(1, 0, 1, 1) = Ch(5, 1);
  C(0, 1, 2, 2) = Ch(5, 2);
  C(1, 0, 2, 2) = Ch(5, 2);
  C(0, 1, 1, 2) = Ch(5, 3);
  C(0, 1, 2, 1) = Ch(5, 3);
  C(1, 0, 1, 2) = Ch(5, 3);
  C(1, 0, 2, 1) = Ch(5, 3);
  C(0, 1, 0, 2) = Ch(5, 4);
  C(0, 1, 2, 0) = Ch(5, 4);
  C(1, 0, 0, 2) = Ch(5, 4);
  C(1, 0, 2, 0) = Ch(5, 4);
  C(0, 1, 0, 1) = Ch(5, 5);
  C(0, 1, 1, 0) = Ch(5, 5);
  C(1, 0, 0, 1) = Ch(5, 5);
  C(1, 0, 1, 0) = Ch(5, 5);

  MatrixXd &K = *new MatrixXd(N * 3, N * 3);

  K.setZero();

  #pragma omp parallel for
  for(int n = 0; n < N; n++)
    for(int m = 0; m < N; m++)
      for(int i = 0; i < 3; i++)
        for(int k = 0; k < 3; k++) {
          double total = 0.0;
              
            for(int j = 0; j < 3; j++)
              for(int l = 0; l < 3; l++)
                total = total + C(i, j, k, l) * dp(n, m, j, l);

          K(n * 3 + i, m * 3 + k) = total;
        }

  *K_ = &K;

  MatrixXd &M = *new MatrixXd(N * 3, N * 3);

  M.setZero();

  for(int n = 0; n < N; n++) {
    for(int m = 0; m < N; m++) {
      M(n * 3 + 0, m * 3 + 0) = density * pv(n, m);
      M(n * 3 + 1, m * 3 + 1) = density * pv(n, m);
      M(n * 3 + 2, m * 3 + 2) = density * pv(n, m);
    }
  }

  *M_ = &M;
}

#endif

