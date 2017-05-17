#include <iostream>

#include <stan/math.hpp>

// Stan function
namespace rus_namespace {
#include "polybasis.hpp"
#include "mechanics.hpp"

  using namespace Eigen;
  using namespace stan::math;

  // Build lookup tables, this is called once in the transformed data block
  template <typename T1__, typename T2__, typename T3__, typename T4__>
  Matrix<typename boost::math::tools::promote_args<T1__, T2__, T3__, T4__>::type, Dynamic, 1>
  mech_init(const int& P,
            const T1__& X,
            const T2__& Y,
            const T3__& Z,
            const T4__& density, std::ostream* pstream__) {
    Matrix<typename boost::math::tools::promote_args<T1__, T2__, T3__, T4__>::type, Dynamic, 1> lookup;
    
    buildBasis(P, X, Y, Z, density, lookup);

    return lookup;
  }

  // This big block of code I just stole from Will Lenthe
  template <typename T>
  Matrix<T, Dynamic, 1>
  ax2qu(const Matrix<T, Dynamic, 1> &ax) {
    T s = sin(ax[3] / 2.0);

    Matrix<T, Dynamic, 1> qu(4);
    
    qu[0] = cos(ax[3] / 2.0);
    qu[1] = s * ax[0];
    qu[2] = s * ax[1];
    qu[3] = s * ax[2];

    return qu;
  }

  template <typename T>
  T hoInv(const T &y_) {
    T x = 2.0 * acos(1.0 - y_ / 2.0);

    T y = sqrt(y_);

    T prevErr(std::numeric_limits<double>::max());

    for(int i = 0; i < 7; i++) { // converges within 6 calculation for all values test within domain
      T fx = pow(0.75 * (x - sin(x)), 1.0 / 3.0);

      T delta = fx - y;

      T err = abs(delta);

      if(0.0 == value_of(delta) || err == prevErr) // no error or flipping between +/- v
        return x;

      x -= 4.0 * fx * fx * delta / (1.0 - cos(x));

      if(err > prevErr) // flipping between +v / -2v (return )
        return x;

      prevErr = err;
    }

    throw std::domain_error("failed to invert ((3/4)*(x-sin(x)))^(2/3)");

    return T(0);
  }

  template <typename T>
  Matrix<T, Dynamic, 1>
  ho2ax(const Matrix<T, Dynamic, 1> &ho) {
    T mag2 = ho.squaredNorm();

    T theta = hoInv(mag2);

    Matrix<T, Dynamic, 1> ax(4);

    ax(0) = ho(0) / sqrt(mag2);
    ax(1) = ho(1) / sqrt(mag2);
    ax(2) = ho(2) / sqrt(mag2);
    ax(3) = theta;

    return ax;
  }

  template <typename T>
  Matrix<T, Dynamic, 1>
  cu2ho(const Matrix<T, Dynamic, 1> &cu) {
    //if max([abs(i) for i in cu]) > cuA() / 2.0 + 7.0 * epsilon:
    //raise ValueError("'%f' lies outside the cube of side length pi^(2/3)" % cu)

    T maxX = max(cu);
    Matrix<T, Dynamic, 1> ho(3);

    if(maxX == cu(2)) {
      ho = cu;
    } else if(maxX == cu(0)) {
      ho(0) = cu(1);
      ho(1) = cu(2);
      ho(2) = cu(0);
    } else if(maxX == cu(1)) {
      ho(0) = cu(2);
      ho(1) = cu(0);
      ho(2) = cu(1);
    }

    //operation M1
    for(size_t i = 0; i < 3; i++)
      ho(i) = ho(i) * pow(M_PI / 6.0, 1.0 / 6.0);

    //operation M2
    bool swapped = false;

    if(abs(ho(0).val()) > abs(ho[1])) {
      swapped = true;
      std::swap(ho(0), ho(1));
    }

    T theta = (M_PI * ho(0)) / (12.0 * ho(1));

    T k = sqrt(3.0 / M_PI) * pow(2.0, 0.75) * ho(1) / sqrt(sqrt(2.0) - cos(theta));
      
    ho(0) = sqrt(2.0) * sin(theta) * k;
    ho(1) = (sqrt(2.0) * cos(theta) - 1.0) * k;
      
    if(swapped)
      std::swap(ho(0), ho(1));

    // operation M3
    k = ho(0) * ho(0) + ho(1) * ho(1);

    for(size_t i = 0; i < 2; i++)
      ho(i) = ho(i) * sqrt(1.0 - M_PI * k / (24.0 * ho(2) * ho(2)));

    ho(2) = sqrt(6.0 / M_PI) * ho(2) - k * sqrt(M_PI / 24.0) / ho(2);

    Matrix<T, Dynamic, 1> hot;    

    if(maxX == cu(2)) {
      hot = ho;
    } else if(maxX == cu(0)) {
      hot(0) = ho(2);
      hot(1) = ho(0);
      hot(2) = ho(1);
    } else if(maxX == cu(1)) {
      hot(0) = ho(1);
      hot(1) = ho(2);
      hot(2) = ho(0);
    }

    return hot;
  }

  template <typename T>
  Matrix<T, Dynamic, 1>
  cu2qu(const Matrix<T, Dynamic, 1> &cu, std::ostream *pstream__) {
    return ax2qu(ho2ax(cu2ho(cu)));
  }

  /*template <typename T0, typename T1>
  Matrix<typename boost::math::tools::promote_args<T0, T1>::type, Dynamic, Dynamic>
  mech_rotate(const Matrix<T0, Dynamic, Dynamic>& C,
              const Matrix<T1, Dynamic, 1>& q, std::ostream* pstream__) {

    const T1& w = q(0);
    const T1& x = q(1);
    const T1& y = q(2);
    const T1& z = q(3);

    Matrix<T1, 3, 3> Q;
    Matrix<T1, 6, 6> K;

    Q << w * w - (y * y + z * z) + x * x, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (y * x + w * z), w * w - (x * x + z * z) + y * y, 2.0 * (y * z - w * x),
      2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w * w - (x * x + y * y) + z * z;

    Q = Q.transpose().eval();

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
    }*/

  // Rotate the 6x6 matrix of stiffness coefficients (modeling the orientation of the crystal
  //   lattice with respect to the sample). q is a passive unit rotation quaternion for
  //   what it's worth.
  //
  // This C++ function is autodiffed magically by Stan. Note the template types
  template <typename T0, typename T1>
  Matrix<typename boost::math::tools::promote_args<T0, T1>::type, Dynamic, Dynamic>
  mech_rotate(const Matrix<T0, Dynamic, Dynamic>& C,
              const Matrix<T1, Dynamic, 1>& q, std::ostream* pstream__) {
    const T1& w = q(0);
    const T1& x = q(1);
    const T1& y = q(2);
    const T1& z = q(3);

    Matrix<T1, 3, 3> Q;
    Matrix<T1, 6, 6> K;

    Q << w * w - (y * y + z * z) + x * x, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (y * x + w * z), w * w - (x * x + z * z) + y * y, 2.0 * (y * z - w * x),
      2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w * w - (x * x + y * y) + z * z;

    Q = Q.transpose().eval();

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

  // Compute the resonance frequencies given the parameters
  //
  // We won't be able to use Stan's autodiff here, so we'll have to define the necessary specializations
  template <typename T1__, typename T2__>
  Matrix<typename boost::math::tools::promote_args<T1__, T2__>::type, Dynamic, 1>
  mech_rus(const int& P, const int& N,
       const Matrix<T1__, Dynamic, 1>& lookup,
       const Matrix<T2__, Dynamic, Dynamic>& C, std::ostream* pstream__);

  // This is the function with the Magic Custom Autodiff. It takes in some stan::math::var s
  //   and spits out some more vars with gradient information embedded in them for the backwards autodiff.
  //   The Stan paper has a good description of how this works https://arxiv.org/abs/1509.07164
  template<>
  inline Matrix<var, Dynamic, 1>
  mech_rus(const int &P, const int& N, const Matrix<double, Dynamic, 1>& lookup, // Constant data
           const Matrix<var, Dynamic, Dynamic>& C, // Parameters
           std::ostream *stream) {
    Matrix<double, Dynamic, 1> freqs(N, 1);

    Matrix<double, Dynamic, 21> dfreqsdCij(N, 21);

    Matrix<double, 6, 6> C_;

    if(C.rows() != 6)
      throw std::runtime_error("Compliance matrix must have exactly 6 rows!");

    if(C.cols() != 6)
      throw std::runtime_error("Compliance matrix must have exactly 6 columns!");
    
    for(int i = 0; i < 6; i++)
      for(int j = 0; j < 6; j++) {
        C_(i, j) = value_of(C(i, j));
      }

    LLT< Matrix<double, 6, 6> > llt = C_.llt();
    if(llt.info() == Eigen::NumericalIssue)
      throw std::runtime_error("Compliance matrix (C) possibly non semi-positive definite!");
    
    //double tmp = omp_get_wtime();
    // This is the big custom function
    mechanics(C_, // Params
              P, lookup, N, // Ref data
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

  // I don't believe this gets called in Vanilla HMC, but cmdStan wanted a specialization
  //   Of the function above that works with doubles. This one does not have gradient information.
  template<>
  inline Matrix<double, Dynamic, 1>
  mech_rus(const int &P, const int& N, const Matrix<double, Dynamic, 1>& lookup, // Constant data
           const Matrix<double, Dynamic, Dynamic>& C, // Parameters
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
        C_(i, j) = C(i, j);
      }

    LLT< Matrix<double, 6, 6> > llt = C_.llt();
    if(llt.info() == Eigen::NumericalIssue)
      throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    
    //double tmp = omp_get_wtime();
    mechanics(C_, // Params
              P, lookup, N, // Ref data
              freqs, // Output
              dfreqsdCij); // Gradients

    return freqs;
  }
}
