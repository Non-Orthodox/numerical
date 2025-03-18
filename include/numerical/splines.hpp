#ifndef NUMERICAL_INCLUDE_SPLINES_HPP
#define NUMERICAL_INCLUDE_SPLINES_HPP

#include "polynomials.hpp"


template<typename Float = double, std::size_t Dim = 1>
class CubicHermitePolynomial
{
private:
  PolynomialVector<3,Dim,Float> polynomial_;
  Float t0_;
  Float tf_;
  Float time_factor_; // 1.0 / (t_f - t_i)

public:
  typedef Eigen::Vector<Float,Dim> Vec;

  CubicHermitePolynomial() {}
  CubicHermitePolynomial(const Vec& p0, const Vec& m0, const Float t0, const Vec& p1, const Vec& m1, const Float t1)
  { Generate(p0, m0, t0, p1, m1, t1); }

  void Generate(const Vec& p0, const Vec& m0, const Float t0, const Vec& p1, const Vec& m1, const Float t1)
  {
    polynomial_.SetTerm(0, p0);
    polynomial_.SetTerm(1, m0);
    polynomial_.SetTerm(2, (-3.0*p0) + (3.0*p1) - (2.0*m0) - m1);
    polynomial_.SetTerm(3, (2.0*p0) - (2.0*p1) + m0 + m1);

    t0_ = t0;
    tf_ = t1;
    time_factor_ = 1.0 / (t1 - t0);
  } 

  auto operator()(const Float& arg) const
  { 
    assert((arg >= t0_) && (arg <= tf_));
    return polynomial_((arg - t0_) * time_factor_);
  }

  Eigen::Vector<Float,2> TimeInterval() const { return {t0_,tf_}; }
  Float InitialTime() const { return t0_; }
  Float FinalTime() const { return tf_; }

  // Eigen::Vector<Float,2> ArcLengthDerivative(Eigen::Vector<Float,2> vec)
  // {
  //   Eigen::Vector<Float,2> result;
  //   result(0) = 1.0;
  //   result(1) = Derivative<1>(vec(0)).norm();
  //   return result;
  // }

  // Float ArcLength(Float arg)
  // {
  //   Float t = (arg - t0_) * time_factor_;
  //   Float dt = (t1 - t0) / 100.0;
  //   Eigen::Vector<Float,2> s = Eigen::Vector<Float,2>::Zero();

  //   // make function for arc length derivative
  //   
  //   while(s(0) < t)
  //   {
  //     if ((s(0)+dt) > t) {
  //       s = RK4<Dim,Float>(s, std::bind(&CubicHermitePolynomial<Float,Dim>::ArcLengthDerivative, 
  //         this, std::placeholders::_1), t-s(0));
  //       break;
  //     }
  //     s = RK4<Dim,Float>(s, std::bind(&CubicHermitePolynomial<Float,Dim>::ArcLengthDerivative, 
  //       this, std::placeholders::_1), dt);
  //   }
  //   return s(1);
  // }

};

#endif
