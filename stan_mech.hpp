#include <iostream>

#include <stan/math.hpp>

// Stan function
namespace rus_namespace {
#include "polybasis.hpp"
#include "mechanics.hpp"

  using namespace Eigen;
  using namespace stan::math;

  int bilayer_lookup_size(const int& IN,
                   const int& JN,
                   const int& KN, std::ostream* pstream__) {
    return computeBilayerSize(IN, JN, KN);
  }

template <typename T3__, typename T4__, typename T5__, typename T6__, typename T7__>
Eigen::Matrix<typename boost::math::tools::promote_args<T3__, T4__, T5__, T6__, typename boost::math::tools::promote_args<T7__>::type>::type, Eigen::Dynamic, 1>
bilayer_init(const int& IN,
	     const int& JN,
	     const int& layer_index,
	     const T3__& X,
	     const T4__& Y,
	     const Eigen::Matrix<T5__, Eigen::Dynamic,1>& Zs,
	     const T6__& bulk_density,
	     const T7__& layer_density,
	     std::ostream* pstream__) {
    return buildBilayerBasis(IN, JN, layer_index, X, Y, Zs, bulk_density, layer_density);
  }

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

      T err = stan::math::abs(delta);

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
    if(cu.array().abs().matrix().maxCoeff() > 1.0725146985555127)
      throw std::domain_error("element of cu lies outside the range (-pi^(2/3), pi^(2/3))");
    
    typename Matrix<T, Dynamic, 1>::Index p;
    cu.array().abs().matrix().maxCoeff(&p);
    Matrix<T, Dynamic, 1> ho(3);

    if(p == 2) {
      ho = cu;
    } else if(p == 0) {
      ho(0) = cu(1);
      ho(1) = cu(2);
      ho(2) = cu(0);
    } else if(p == 1) {
      ho(0) = cu(2);
      ho(1) = cu(0);
      ho(2) = cu(1);
    }

    //operation M1
    for(size_t i = 0; i < 3; i++)
      ho(i) = ho(i) * pow(M_PI / 6.0, 1.0 / 6.0);

    //operation M2
    bool swapped = false;

    if(stan::math::abs(ho(0)) > stan::math::abs(ho[1])) {
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

    Matrix<T, Dynamic, 1> hot(3);

    if(p == 2) {
      hot = ho;
    } else if(p == 0) {
      hot(0) = ho(2);
      hot(1) = ho(0);
      hot(2) = ho(1);
    } else if(p == 1) {
      hot(0) = ho(1);
      hot(1) = ho(2);
      hot(2) = ho(0);
    }

    return hot;
  }

  template <typename T0__>
  Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, Eigen::Dynamic,1>
  cu2qu(const Eigen::Matrix<T0__, Eigen::Dynamic,1>& cu, std::ostream* pstream__) {
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

  // If mech_rus was passed vars, we need to package up the gradients for the output
  inline Matrix<var, Dynamic, 1> build_output(const Matrix<var, Dynamic, 1>& C,
                                              const Matrix<double, Dynamic, 1>& freqs,
                                              const Matrix<double, Dynamic, Dynamic>& dfreqsdCij) {
    int N = freqs.size();
    
    Matrix<var, Dynamic, 1> retval(N);

    vari** params = ChainableStack::instance_->memalloc_.alloc_array<vari *>(C.size());

    for(int i = 0; i < C.size(); i++) {
      params[i] = C(i).vi_;
    }

    for(int i = 0; i < N; i++) {
      double* gradients = ChainableStack::instance_->memalloc_.alloc_array<double>(C.size());

      for(int j = 0; j < C.size(); j++)
        gradients[j] = dfreqsdCij(i, j);
        
      retval(i) = var(new stored_gradient_vari(freqs(i), C.size(), params, gradients));
    }
  
    return retval;
  }

  // If mech_rus was passed only doubles, then we don't need to fuss with the gradients
  inline Matrix<double, Dynamic, 1> build_output(const Matrix<double, Dynamic, 1>& C,
                                                 const Matrix<double, Dynamic, 1>& freqs,
                                                 const Matrix<double, Dynamic, Dynamic>& dfreqsdCij) {
    return freqs;
  }

  inline void flatten(int Ksize,
                      const Matrix<double, Dynamic, 1>& lookup, // Constant data
                      const Matrix<var, Dynamic, Dynamic>& C, // Parameters
                      VectorXd& lookup_,
                      Matrix<var, Dynamic, 1>& C_) {
    std::map<vari *, var> unique;
    std::map<vari *, VectorXd> lookup_map;
    int ij = 0;
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < i + 1; j++) {
        if(unique.find(C(i, j).vi_) == unique.end()) {
          unique[C(i, j).vi_] = C(i, j);
          lookup_map[C(i, j).vi_] = VectorXd::Zero(Ksize);
        }
        for(int k = 0; k < Ksize; k++) {
          lookup_map[C(i, j).vi_](k) += lookup[ij * Ksize + k];
        }
        ij += 1;
      }
    }
    C_.resize(unique.size());
    lookup_.resize(Ksize * unique.size());
    
    int i = 0;
    for(auto&& it : unique) {
      C_(i) = it.second;
      for(int j = 0; j < Ksize; j++) {
        lookup_[i * Ksize + j] = lookup_map[it.first](j);
      }
      i += 1;
    }
  }

  inline void flatten(int Ksize,
                      const Matrix<double, Dynamic, 1>& lookup, // Constant data
                      const Matrix<double, Dynamic, Dynamic>& C, // Parameters
                      VectorXd& dfreqsdCij,
                      VectorXd& C_) {
    dfreqsdCij = lookup;
    C_.resize(21);
    int ij = 0;
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < i + 1; j++) {
        C_(ij) = C(i, j);
        ij++;
      }
    }
  }
  template<typename T1, typename T2>
  inline Matrix<typename boost::math::tools::promote_args<T1, T2>::type, Dynamic, 1>
  mech_rus(const int& P,
	   const int& N,
           const Matrix<T1, Dynamic, 1>& lookup, // Constant data
           const Matrix<T2, Dynamic, Dynamic>& C, // Parameters
           std::ostream *stream) {
    throw std::runtime_error("You're using an old model file. mech_rus doesn't take a 'P' argument anymore");
  }

  // Compute the resonance frequencies given the parameters
  //
  // We won't be able to use Stan's autodiff here, so we'll have to define the
  // necessary specializations
  //
  // This is the function takes in some stan::math::var s and spits out some
  // more vars with gradient information embedded in them for the backwards autodiff.
  //
  // The Stan paper has a good description of how this works
  // https://arxiv.org/abs/1509.07164
  template<typename T1, typename T2>
  inline Matrix<typename boost::math::tools::promote_args<T1, T2>::type, Dynamic, 1>
  mech_rus(const int& N,
           const Matrix<T1, Dynamic, 1>& lookup, // Constant data
           const Matrix<T2, Dynamic, Dynamic>& C, // Parameters
           std::ostream *stream) {
    if(lookup.size() % 21 != 0)
      throw std::runtime_error("lookup.size() must be a multiple of 21!");

    if(C.rows() != 6)
      throw std::runtime_error("Compliance matrix must have exactly 6 rows!");

    if(C.cols() != 6)
      throw std::runtime_error
        ("Compliance matrix must have exactly 6 columns!");

    LLT< Matrix<double, Dynamic, Dynamic> > llt = value_of(C).llt();
    if(llt.info() == Eigen::NumericalIssue)
      throw std::domain_error
        ("Compliance matrix (C) non semi-positive definite!");

    int Ksize = lookup.size() / 21;

    VectorXd lookup_;
    Matrix<T2, Dynamic, 1> C_;

    flatten(Ksize, lookup, C, lookup_, C_);

    Matrix<double, Dynamic, 1> freqs(N, 1);
    MatrixXd dfreqsdCij(N, C_.size());
    
    //double tmp = omp_get_wtime();
    // This is the big custom function
    mechanics(value_of(C_), // Params
              lookup_, N, // Ref data
              freqs, // Output
              dfreqsdCij); // Gradients

    // Package up output (this will be different depending on the template
    // types of the inputs)
    return build_output(C_, freqs, dfreqsdCij);
  }

  inline void bilayer_flatten(int Ksize,
                              const Matrix<double, Dynamic, 1>& lookup, // Constant data
                              const Matrix<var, Dynamic, Dynamic>& C1, // Parameters
                              const Matrix<var, Dynamic, Dynamic>& C2,
                              VectorXd& lookup_,
                              Matrix<var, Dynamic, 1>& C_) {
    std::map<vari *, var> unique;
    std::map<vari *, VectorXd> lookup_map;
    int ij = 0;
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < i + 1; j++) {
        if(unique.find(C1(i, j).vi_) == unique.end()) {
          unique[C1(i, j).vi_] = C1(i, j);
          lookup_map[C1(i, j).vi_] = VectorXd::Zero(Ksize);
        }
        for(int k = 0; k < Ksize; k++) {
          lookup_map[C1(i, j).vi_](k) += lookup[ij * Ksize + k];
        }
        ij += 1;
      }
    }
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < i + 1; j++) {
        if(unique.find(C2(i, j).vi_) == unique.end()) {
          unique[C2(i, j).vi_] = C2(i, j);
          lookup_map[C2(i, j).vi_] = VectorXd::Zero(Ksize);
        }
        for(int k = 0; k < Ksize; k++) {
          lookup_map[C2(i, j).vi_](k) += lookup[ij * Ksize + k];
        }
        ij += 1;
      }
    }
    C_.resize(unique.size());
    lookup_.resize(Ksize * unique.size());
    
    int i = 0;
    for(auto&& it : unique) {
      C_(i) = it.second;
      for(int j = 0; j < Ksize; j++) {
        lookup_[i * Ksize + j] = lookup_map[it.first](j);
      }
      i += 1;
    }
  }

  inline void bilayer_flatten(int Ksize,
                              const Matrix<double, Dynamic, 1>& lookup, // Constant data
                              const Matrix<double, Dynamic, Dynamic>& C1, // Parameters
                              const Matrix<double, Dynamic, Dynamic>& C2,
                              VectorXd& dfreqsdCij,
                              VectorXd& C_) {
    dfreqsdCij = lookup;
    C_.resize(21 * 2);
    int ij = 0;
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < i + 1; j++) {
        C_(ij) = C1(i, j);
        ij += 1;
      }
    }
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < i + 1; j++) {
        C_(ij) = C2(i, j);
        ij += 1;
      }
    }
  }

  // Compute the resonance frequencies given the parameters
  //
  // We won't be able to use Stan's autodiff here, so we'll have to define the
  // necessary specializations
  //
  // This is the function takes in some stan::math::var s and spits out some
  // more vars with gradient information embedded in them for the backwards autodiff.
  //
  // The Stan paper has a good description of how this works
  // https://arxiv.org/abs/1509.07164
  template<typename T1, typename T2>
  inline Matrix<typename boost::math::tools::promote_args<T1, T2>::type, Dynamic, 1>
  bilayer_rus(const int& N,
              const Matrix<T1, Dynamic, 1>& lookup, // Constant data
              const Matrix<T2, Dynamic, Dynamic>& C1, // Parameters
              const Matrix<T2, Dynamic, Dynamic>& C2, // Parameters
              std::ostream* stream) {
    if(lookup.size() % (36 * 2) != 0)
      throw std::runtime_error("lookup.size() must be a multiple of 21!");

    if(C1.rows() != 6)
      throw std::runtime_error("Compliance matrix 1 must have exactly 6 rows!");

    if(C2.rows() != 6)
      throw std::runtime_error("Compliance matrix 2 must have exactly 6 rows!");

    if(C1.cols() != 6)
      throw std::runtime_error
        ("Compliance matrix 1 must have exactly 6 columns!");

    if(C2.cols() != 6)
      throw std::runtime_error
        ("Compliance matrix 2 must have exactly 6 columns!");

    LLT< Matrix<double, Dynamic, Dynamic> > llt1 = value_of(C1).llt();
    if(llt1.info() == Eigen::NumericalIssue)
      throw std::domain_error
        ("Compliance matrix (C1) non semi-positive definite!");

    LLT< Matrix<double, Dynamic, Dynamic> > llt2 = value_of(C2).llt();
      if(llt2.info() == Eigen::NumericalIssue)
        throw std::domain_error
          ("Compliance matrix (C2) non semi-positive definite!");

    int Ksize = lookup.size() / (21 * 2);

    VectorXd lookup_ = lookup;
    Matrix<T2, Dynamic, 1> C_(C1.size() + C2.size());

    for(int i = 0; i< C1.size(); i++) {
      C_(i) = C1(i);
      C_(36 + i) = C2(i);
    }

    //bilayer_flatten(Ksize, lookup, C1, C2, lookup_, C_);

    Matrix<double, Dynamic, 1> freqs(N, 1);
    MatrixXd dfreqsdCij(N, C_.size());
    
    //double tmp = omp_get_wtime();
    // This is the big custom function
    mechanics(value_of(C_), // Params
              lookup_, N, // Ref data
              freqs, // Output
              dfreqsdCij); // Gradients

    // Package up output (this will be different depending on the template
    // types of the inputs)
    return build_output(C_, freqs, dfreqsdCij);
  }
}
