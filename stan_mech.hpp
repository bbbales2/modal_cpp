#include <iostream>

#include <stan/math.hpp>

// Stan function
namespace test_model_namespace {
#include "polybasis.hpp"
#include "mechanics.hpp"

  using namespace Eigen;
  using namespace stan::math;

  inline Matrix<var, Dynamic, Dynamic>
  rotate(const Matrix<var, Dynamic, Dynamic>& C, const Matrix<var, Dynamic, 1>& q) {
    const var& w = q(0);
    const var& x = q(1);
    const var& y = q(2);
    const var& z = q(3);

    Matrix<var, 3, 3> Q;
    Matrix<var, 6, 6> K;

    Q << w * w - (y * y + z * z) + x * x, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (y * x + w * z), w * w - (x * x + z * z) + y * y, 2.0 * (y * z - w * x),
      2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w * w - (x * x + y * y) + z * z;

    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
        K(i, j) = Q(i, j) * Q(i, j);
        K(i, j + 3) = Q(i, (j + 1) % 3) * Q(i, (j + 2) % 3);
        K(i + 3, j) = Q((i + 1) % 3, j) * Q((i + 2) % 3, j);
        K(i + 3, j + 3) = Q((i + 1) % 3, (j + 1) % 3) * Q((i + 2) % 3, (j + 2) % 3) + Q((i + 1) % 3, (j + 2) % 3) * Q((i + 2) % 3, (j + 1) % 3);
      }
    }

    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++)
        K(i, j + 3) *= 2.0;

    return K * C * K.transpose();
  }
  
  template <typename T1__, typename T2__, typename T3__>
  Eigen::Matrix<typename boost::math::tools::promote_args<T1__, T2__, T3__>::type, Eigen::Dynamic,1>
  mech(const int& N,
       const Eigen::Matrix<T1__, Eigen::Dynamic, 1>& dp,
       const Eigen::Matrix<T2__, Eigen::Dynamic, Eigen::Dynamic>& pv,
       const Eigen::Matrix<T3__, Eigen::Dynamic, Eigen::Dynamic>& C, std::ostream* pstream__);
  
  template<>
  inline Matrix<var, Dynamic, 1>
  mech(const int& N, const dpT& dp, const pvT& pv, // Constant data
       const Matrix<var, Dynamic, Dynamic>& C, // Parameters
       std::ostream *stream) {

    Matrix<double, Dynamic, 1> freqs(N, 1);

    Matrix<double, Dynamic, 21> dfreqsdCij(N, 21);

    Matrix<double, 6, 6> C_;

    if(C.rows() != 6)
      std::cout << "WRONG NUMBER OF ROWS IN COMPLIANCE MATRIX" << std::endl;

    if(C.cols() != 6)
      std::cout << "WRONG NUMBER OF COLS IN COMPLIANCE MATRIX" << std::endl;
    
    for(int i = 0; i < 6; i++)
      for(int j = 0; j < 6; j++) {
        C_(i, j) = C(i, j).vi_->val_;

        //std::cout << C(j, i) << " " << C(i, j) << std::endl;
        
        if(C(i, j) != C(j, i))
          std::cout << "MATRIX IS NOT POSITIVE DEFINITE" << std::endl;
      }
    
    double tmp = omp_get_wtime();
    mechanics(C_, // Params
              dp, pv, N, // Ref data
              freqs, // Output
              dfreqsdCij); // Gradients

    Matrix<var, Dynamic, 1> retval(N);
  
    vari** params = ChainableStack::memalloc_.alloc_array<vari *>(21);

    int ij = 0;
    for(int i = 0; i < 6; i++)
      for(int j = 0; j < i + 1; j++) {
        params[ij] = C(i, j).vi_;

        ij++;
      }

    for(int i = 0; i < N; i++) {
      double* gradients = ChainableStack::memalloc_.alloc_array<double>(21);

      for(int ij = 0; ij < 21; ij++)
        gradients[ij] = dfreqsdCij(i, ij);
        
      retval(i) = var(new stored_gradient_vari(freqs(i), 21, params, gradients));
    }
  
    return retval;
  }
}
