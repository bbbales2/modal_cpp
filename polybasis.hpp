#include <cmath>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

double polyint(int n, int m, int l)
{
  double xtmp, ytmp;

  if(n < 0 | m < 0 | l < 0 | n % 2 > 0 | m % 2 > 0 | l % 2 > 0)
    return 0.0;
  
  xtmp = 2.0 * pow(0.5, n + 1);
  ytmp = 2.0 * xtmp * pow(0.5, m + 1);

  return 2.0 * pow(0.5, l + 1) * ytmp / ((n + 1) * (m + 1) * (l + 1));
}

void build(int N, double X, double Y, double Z, Eigen::Tensor<double, 4> **dp_, Eigen::Tensor<double, 2> **pv_) {
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
/*
cpdef buildRot(C, w, x, y, z):
    # Code stolen from Will Lenthe
    K = numpy.zeros((6, 6))
    dKdQ = numpy.zeros((6, 6, 3, 3))

    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    dQdw = numpy.array([[2 * w, -2.0 * z, 2.0 * y],
                        [2.0 * z, 2 * w, -2.0 * x],
                        [-2.0 * y, 2.0 * x, 2 * w]])

    dQdx = numpy.array([[2 * x, 2.0 * y, 2.0 * z],
                        [2.0 * y, -2.0 * x, -2.0 * w],
                        [2.0 * z, 2.0 * w, -2.0 * x]])

    dQdy = numpy.array([[-2 * y, 2 * x, 2 * w],
                        [2 * x, 2 * y, 2 * z],
                        [-2 * w, 2 * z, -2 * y]])

    dQdz = numpy.array([[-2 * z, -2 * w, 2 * x],
                        [2 * w, -2 * z, 2 * y],
                        [2 * x, 2 * y, 2 * z]])

    for i in range(3):
        for j in range(3):
            dKdQ[i, j, i, j] = 2.0 * Q[i, j]
            dKdQ[i, j + 3, i, (j + 1) % 3] = Q[i, (j + 2) % 3]
            dKdQ[i, j + 3, i, (j + 2) % 3] = Q[i, (j + 1) % 3]
            dKdQ[i + 3, j, (i + 1) % 3, j] = Q[(i + 2) % 3, j]
            dKdQ[i + 3, j, (i + 2) % 3, j] = Q[(i + 1) % 3, j]
            dKdQ[i + 3, j + 3, (i + 1) % 3, (j + 1) % 3] = Q[(i + 2) % 3, (j + 2) % 3]
            dKdQ[i + 3, j + 3, (i + 2) % 3, (j + 2) % 3] = Q[(i + 1) % 3, (j + 1) % 3]
            dKdQ[i + 3, j + 3, (i + 1) % 3, (j + 2) % 3] = Q[(i + 2) % 3, (j + 1) % 3]
            dKdQ[i + 3, j + 3, (i + 2) % 3, (j + 1) % 3] = Q[(i + 1) % 3, (j + 2) % 3]

            K[i][j] = Q[i][j] * Q[i][j]
            K[i][j + 3] = Q[i][(j + 1) % 3] * Q[i][(j + 2) % 3]
            K[i + 3][j] = Q[(i + 1) % 3][j] * Q[(i + 2) % 3][j]
            K[i + 3][j + 3] = Q[(i + 1) % 3][(j + 1) % 3] * Q[(i + 2) % 3][(j + 2) % 3] + Q[(i + 1) % 3][(j + 2) % 3] * Q[(i + 2) % 3][(j + 1) % 3]

    for i in range(3):
        for j in range(3):
            K[i][j + 3] *= 2.0
            dKdQ[i][j + 3] *= 2.0

    Crot = K.dot(C.dot(K.T))
    dCrotdQ = numpy.zeros((6, 6, 3, 3))
    for i in range(3):
        for j in range(3):
            dCrotdQ[:, :, i, j] = dKdQ[:, :, i, j].dot(C.dot(K.T)) + K.dot(C.dot(dKdQ[:, :, i, j].T))

    dCrotdw = numpy.zeros((6, 6))
    dCrotdx = numpy.zeros((6, 6))
    dCrotdy = numpy.zeros((6, 6))
    dCrotdz = numpy.zeros((6, 6))

    for i in range(6):
        for j in range(6):
            dCrotdw[i, j] = (dCrotdQ[i, j] * dQdw).flatten().sum()
            dCrotdx[i, j] = (dCrotdQ[i, j] * dQdx).flatten().sum()
            dCrotdy[i, j] = (dCrotdQ[i, j] * dQdy).flatten().sum()
            dCrotdz[i, j] = (dCrotdQ[i, j] * dQdz).flatten().sum()

    return Crot, dCrotdw, dCrotdx, dCrotdy, dCrotdz, K

cpdef buildRot2(C, w, x, y, z):
    # Code stolen from Will Lenthe
    K = numpy.zeros((6, 6))
    dKdQ = numpy.zeros((6, 6, 3, 3))

    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    dQdw = numpy.array([[2 * w, -2.0 * z, 2.0 * y],
                        [2.0 * z, 2 * w, -2.0 * x],
                        [-2.0 * y, 2.0 * x, 2 * w]])

    dQdx = numpy.array([[2 * x, 2.0 * y, 2.0 * z],
                        [2.0 * y, -2.0 * x, -2.0 * w],
                        [2.0 * z, 2.0 * w, -2.0 * x]])

    dQdy = numpy.array([[-2 * y, 2 * x, 2 * w],
                        [2 * x, 2 * y, 2 * z],
                        [-2 * w, 2 * z, -2 * y]])

    dQdz = numpy.array([[-2 * z, -2 * w, 2 * x],
                        [2 * w, -2 * z, 2 * y],
                        [2 * x, 2 * y, 2 * z]])

    Cv = numpy.zeros((3, 3, 3, 3))

    Cv = Cvoigt(C)

    Crot = numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, Q)

    dCrotdw = numpy.einsum('ip, jq, pqrs, kr, ls', dQdw, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdw, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdw, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdw)

    dCrotdx = numpy.einsum('ip, jq, pqrs, kr, ls', dQdx, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdx, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdx, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdx)

    dCrotdy = numpy.einsum('ip, jq, pqrs, kr, ls', dQdy, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdy, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdy, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdy)

    dCrotdz = numpy.einsum('ip, jq, pqrs, kr, ls', dQdz, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdz, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdz, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdz)

    return Crot, dCrotdw, dCrotdx, dCrotdy, dCrotdz, Q

cpdef inline numpy.ndarray[numpy.double_t, ndim = 4] Cvoigt(numpy.ndarray[numpy.double_t, ndim = 2] Ch):
    cdef numpy.ndarray[numpy.double_t, ndim = 4] C
    cdef int i, j, k, l, n, m

    C = numpy.zeros((3, 3, 3, 3))

    voigt = [[(0, 0)], [(1, 1)], [(2, 2)], [(1, 2), (2, 1)], [(0, 2), (2, 0)], [(0, 1), (1, 0)]]

    for i in range(6):
        for j in range(6):
            for k, l in voigt[i]:
                for n, m in voigt[j]:
                    C[k, l, n, m] = Ch[i, j]
    return C

import cython
from cython cimport parallel
*/

void buildKM(Eigen::Tensor<double, 2> &Ch, Eigen::Tensor<double, 4> &dp, Eigen::Tensor<double, 2> &pv, double density, Eigen::Tensor<double, 2> **K_, Eigen::Tensor<double, 2> **M_) {
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

  Eigen::Tensor<double, 2> &K = *new Eigen::Tensor<double, 2>(N * 3, N * 3);

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

  Eigen::Tensor<double, 2> &M = *new Eigen::Tensor<double, 2>(N * 3, N * 3);

  for(int n = 0; n < N; n++) {
    for(int m = 0; m < N; m++) {
      M(n * 3 + 0, m * 3 + 0) = density * pv(n, m);
      M(n * 3 + 1, m * 3 + 1) = density * pv(n, m);
      M(n * 3 + 2, m * 3 + 2) = density * pv(n, m);
    }
  }

  *M_ = &M;
}
