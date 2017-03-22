#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <stdio.h>

#include "polybasis.hpp"

extern "C" void dsygvx_(int *itype,
                        const char *jobz,
                        const char *range,
                        const char *uplo,
                        int *N,
                        double *A,
                        int *lda,
                        double *B,
                        int *ldb,
                        double *vl,
                        double *vu,
                        int *il,
                        int *iu,
                        double *abstol,
                        int *M,
                        double *W,
                        double *Z,
                        int *ldz,
                        double *work,
                        int *lwork,
                        int *iwork,
                        int *ifail,
                        int *info);

MatrixXd eigsD(MatrixXd &A, MatrixXd &B, int il, int iu)
{
  // Rescaling the matrices a little seems to help the solve go faster, won't hurt the eigenvalues/vecs
  double max = std::min(A.maxCoeff(), B.maxCoeff());

  MatrixXd At = A.transpose().eval() / max;
  MatrixXd Bt = B.transpose().eval() / max;

  il += 1;
  iu += 1;
  
  int itype = 1;
  int N = A.rows();
  int lda = N;
  int ldb = N;
  int ldz = N;
  double abstol = 1e-7;
  int M = 0;
  MatrixXd W(iu - il + 1, 1);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Z(N, iu - il + 1);
  std::vector<int> iwork(5 * N);
  std::vector<int> ifail(N);
  int info = 0;
  double zero = 0.0;
  double workQuery = 0.0;
  int lworkQuery = -1;

  //double tmp = omp_get_wtime();

  // First call computes optimal workspace storage size
  dsygvx_(&itype, "V", "I", "L", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, W.data(), Z.data(), &ldz, &workQuery, &lworkQuery, &iwork[0], &ifail[0], &info);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, workQuery, omp_get_wtime() - tmp);

  int lwork = int(workQuery) + 1;
  std::vector<double> work(lwork);

  // Second call actually computes the eigenvalues and eigenvectors!
  dsygvx_(&itype, "V", "I", "L", &N, At.data(), &lda, Bt.data(), &ldb, &zero, &zero, &il, &iu, &abstol, &M, W.data(), Z.data(), &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info);

  //printf("Info %d, M %d, opt %f, %f\n", info, M, work[0], omp_get_wtime() - tmp);
  
  return W;
}

int main2(int argc, char **argv) {
  int N = 10;

  MatrixXd A(N, N);
  MatrixXd B(N, N);

  //Eigen::Tensor<double, 2, Eigen::ColMajor> &A = *new Eigen::Tensor<double, 2, Eigen::ColMajor>(N, N);

  //A.setRandom();

  //for(int i = 0; i < N; i++) {
  //  for(int j = i; j < N; j++) {
  //    A(j, i) = A(i, j);
  //  }
  //}

  A.setZero();
  B.setZero();

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      if(i == j) {
        A(i, j) = i + 1.0;
        B(i, j) = 2.00;
        printf("%d %d %f %f\n", i, j, B(i, j), A(i, j));
      }
    }
  }

  double tmp = omp_get_wtime();
  MatrixXd eigs = eigsD(A, B, 0, 7);
  printf("Time %f\n", omp_get_wtime() - tmp);
  
  printf("Eigenvalues:\n");
  for(int i = 0; i < eigs.rows(); i++) {
    for(int j = 0; j < eigs.cols(); j++) {
      printf("%e ", eigs(i, j));
    }
    printf("\n");
  }

  /*printf("\nEigenvectors:\n");
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < nev; j++) {
      printf("%f ", Z(i, j));
    }
    printf("\n");
    }*/
}

int main(int argc, char **argv)
{
  double X = 0.007753,
    Y = 0.009057,
    Z = 0.013199;

  double c11 = 1.685;
  double anisotropic = 1.0;
  double c44 = 0.446;
  double c12 = -(c44 * 2.0 / anisotropic - c11);

  double density = 4401.695921;

  Eigen::Tensor<double, 4> *dp;
  Eigen::Tensor<double, 2> *pv;
  
  double tmp = omp_get_wtime();
  buildBasis(14, X, Y, Z, &dp, &pv);
  printf("buildBasis %f\n", omp_get_wtime() - tmp);

  Eigen::Tensor<double, 2> C(6, 6);

  C.setValues({{c11, c12, c12, 0, 0, 0},
               {c12, c11, c12, 0, 0, 0},
               {c12, c12, c11, 0, 0, 0},
               {0, 0, 0, c44, 0, 0},
               {0, 0, 0, 0, c44, 0},
               {0, 0, 0, 0, 0, c44}});

  MatrixXd *K, *M;

  tmp = omp_get_wtime();
  buildKM(C, *dp, *pv, density, &K, &M);
  printf("buildKM %f\n", omp_get_wtime() - tmp);

  tmp = omp_get_wtime();
  MatrixXd *eigs, *evecs;
  eigsD(*K, *M, 6, 59, &eigs, &evecs);
  printf("Time %f\n", omp_get_wtime() - tmp);
  
  /*M->setZero();

  for(int i = 0; i < M->rows(); i++) {
    for(int j = 0; j < M->cols(); j++) {
      if(i == j) {
        (*M)(i, j) = 1.0;
      }
    }
    } */ 
  
  printf("Eigenvalues:\n");
  for(int i = 0; i < eigs.rows(); i++) {
    for(int j = 0; j < eigs.cols(); j++) {
      printf("%e ", eigs(i, j));
    }
    printf("\n");
  }

  /*for(int i = 0; i < 10; i++) {
    for(int j = 0; j < 10; j++) {
      printf("%e ", (*K)(i, j));
    }
    printf("\n");
    }*/
  
  return 0;
}
