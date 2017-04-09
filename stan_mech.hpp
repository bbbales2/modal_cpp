#include <iostream>

#include <stan/math.hpp>

// Stan function
namespace test_model_namespace {
#include "polybasis.hpp"
#include "mechanics.hpp"

  using namespace Eigen;
  using namespace stan::math;

  template <typename T1__, typename T2__, typename T3__, typename T4__, typename T5__>
  Eigen::Matrix<typename boost::math::tools::promote_args<T1__, T2__, T3__, T4__, typename boost::math::tools::promote_args<T5__>::type>::type, Eigen::Dynamic,1>
  mech(const int& N,
       const Eigen::Matrix<T1__, Eigen::Dynamic,1>& dp,
       const Eigen::Matrix<T2__, Eigen::Dynamic,Eigen::Dynamic>& pv,
       const T3__& c11,
       const T4__& a,
       const T5__& c44, std::ostream* pstream__);
  
  template<>
  inline Matrix<var, Dynamic, 1>
  mech(const int& N, const dpT& dp, const pvT& pv, // Constant data
       const var& c11, const var& anisotropic, const var& c44, // Parameters
       std::ostream *stream) {

    Matrix<double, Dynamic, 1> freqs(N),
      dfreqs_dc11(N),
      dfreqs_da(N),
      dfreqs_dc44(N);

    double tmp = omp_get_wtime();
    mechanics(c11.vi_->val_, anisotropic.vi_->val_, c44.vi_->val_, // Params
              dp, pv, N, // Ref data
              freqs, // Output
              dfreqs_dc11, dfreqs_da, dfreqs_dc44); // Gradients

    //if(stream)
    //  (*stream) << omp_get_wtime() - tmp << " " << c11.vi_->val_ << " " << anisotropic.vi_->val_ << " " << c44.vi_->val_ << std::endl;
  
    Matrix<var, Dynamic, 1> retval(N);
  
    vari** params = ChainableStack::memalloc_.alloc_array<vari *>(3);
  
    params[0] = c11.vi_;
    params[1] = anisotropic.vi_;
    params[2] = c44.vi_;

    for(int i = 0; i < N; i++) {
      double* gradients = ChainableStack::memalloc_.alloc_array<double>(3);
    
      gradients[0] = dfreqs_dc11(i);
      gradients[1] = dfreqs_da(i);
      gradients[2] = dfreqs_dc44(i);
        
      retval(i) = var(new stored_gradient_vari(freqs(i), 3, params, gradients));
    }
  
    return retval;
  }

  template <>
  Matrix<double, Dynamic,1>
  mech(const int& N, const dpT& dp, const pvT& pv,
       const double& c11, const double& anisotropic, const double& c44, std::ostream* stream)
  /*inline Matrix<var, Dynamic, 1> mech(const int& N, const dpT& dp, const pvT& pv, // Constant data
    const var& c11, const var& anisotropic, const var& c44, std::ostream *stream = NULL)*/ { // Parameters

    //(*stream) << "Called double func" << std::endl;
      
    Matrix<double, Dynamic, 1> freqs(N),
      dfreqs_dc11(N),
      dfreqs_da(N),
      dfreqs_dc44(N);
    
    double tmp = omp_get_wtime();
    mechanics(c11, anisotropic, c44, // Params
              dp, pv, N, // Ref data
              freqs, // Output
              dfreqs_dc11, dfreqs_da, dfreqs_dc44); // Gradients
    
    //if(stream)
    //  (*stream) << omp_get_wtime() - tmp << std::endl;
    
    Matrix<double, Dynamic, 1> retval(N);
    
    for(int i = 0; i < N; i++)
      retval(i) = freqs(i);
    
    return retval;
  }

  inline Matrix<var, Dynamic, 1>
  mechr(const int& N, const dpT& dp, const pvT& pv, // Constant data
        const var& c11, const var& anisotropic, const var& c44, // Parameters
        const var& w, const var& x, const var& y, const var& z,
        std::ostream *stream) {

    Matrix<double, Dynamic, 1> freqs,
      dfreqs_dc11,
      dfreqs_da,
      dfreqs_dc44,
      dfreqs_dw,
      dfreqs_dx,
      dfreqs_dy,
      dfreqs_dz;

    double tmp = omp_get_wtime();
    mechanicsr(c11.vi_->val_, anisotropic.vi_->val_, c44.vi_->val_, // Params
               w.vi_->val_, x.vi_->val_, y.vi_->val_, z.vi_->val_,
               dp, pv, N, // Ref data
               freqs, // Output
               dfreqs_dc11, dfreqs_da, dfreqs_dc44,
               dfreqs_dw, dfreqs_dx, dfreqs_dy, dfreqs_dz); // Gradients

    //if(stream)
    //  (*stream) << omp_get_wtime() - tmp << " " << c11.vi_->val_ << " " << anisotropic.vi_->val_ << " " << c44.vi_->val_ << std::endl;
  
    Matrix<var, Dynamic, 1> retval(N);
  
    vari** params = ChainableStack::memalloc_.alloc_array<vari *>(7);
  
    params[0] = c11.vi_;
    params[1] = anisotropic.vi_;
    params[2] = c44.vi_;
    params[3] = w.vi_;
    params[4] = x.vi_;
    params[5] = y.vi_;
    params[6] = z.vi_;

    for(int i = 0; i < N; i++) {
      double* gradients = ChainableStack::memalloc_.alloc_array<double>(7);
    
      gradients[0] = dfreqs_dc11(i);
      gradients[1] = dfreqs_da(i);
      gradients[2] = dfreqs_dc44(i);
      gradients[3] = dfreqs_dw(i);
      gradients[4] = dfreqs_dx(i);
      gradients[5] = dfreqs_dy(i);
      gradients[6] = dfreqs_dz(i);
        
      retval(i) = var(new stored_gradient_vari(freqs(i), 7, params, gradients));
    }
  
    return retval;
  }
}
