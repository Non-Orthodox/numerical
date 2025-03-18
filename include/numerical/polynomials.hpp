#ifndef NUMERICAL_INCLUDE_POLYNOMIALS_HPP
#define NUMERICAL_INCLUDE_POLYNOMIALS_HPP

#include <iostream>
#include <vector>
#include <type_traits>

#include <Eigen/Dense>


template<typename Float = double>
Float MonomialDerivative(const Float& coeff, const std::size_t power, const std::size_t derivative_order)
{
  Float result = coeff;
  if (derivative_order == 0) {
    return result;
  }
  if (derivative_order > power) {
    return static_cast<Float>(0);
  }
  for (std::size_t i = 0; i < derivative_order; i++) {
    result *= static_cast<Float>(power - i);
  }
  return result;
}


template<typename Float, std::size_t Dim>
Eigen::Vector<Float, Dim> MonomialVectorDerivative(const Eigen::Ref<const Eigen::Vector<Float,Dim>>& coeffs,
    const std::size_t power, const std::size_t derivative_order)
{
  Eigen::Vector<Float,Dim> result = coeffs;
  if (derivative_order == 0) {
    return result;
  }
  if (derivative_order > power) {
    return Eigen::Vector<Float,Dim>::Zero();
  }
  for (std::size_t i = 0; i < derivative_order; i++) {
    result *= static_cast<Float>(power - i);
  }
  return result;
}


//TODO make it so PolyOrder and RangeDim can be dynamic (-1)
template<std::size_t PolyOrder, std::size_t RangeDim = 1, typename Float = double>
class Polynomial
{
public:
  typedef Eigen::Vector<Float,PolyOrder+1> PVec;
  typedef typename
    std::conditional_t<RangeDim==1, Float, Eigen::Vector<Float,RangeDim>> RVec;
  typedef Eigen::Matrix<Float, RangeDim, PolyOrder+1> Mat;

protected:
  Mat coeffs_; // First index is power of 0, second is power 1, etc. One row for each input.

public:
  Polynomial() {}
  Polynomial(Mat coeffs) : coeffs_{coeffs} {}
  ~Polynomial() {}

  void SetCoeffs(const Mat coeffs) { coeffs_ = coeffs; }
  
  void SetCoeffs(const PVec coeffs)
  {
    static_assert(RangeDim==1,"Polynomial<PolyOrder,RangeDim,Float>::SetCoeffs(Eigen::Vector<Float,RangeDim>) called when RangeDim != 1");
    coeffs_ = coeffs.transpose();
  }

  void SetZero() { coeffs_ = Mat::Zero(); }

  // void SetTerm(const std::size_t col_index, const Eigen::Ref<const RVec>& args)
  // { coeffs_.col(col_index) = args; }

  void SetTerm(const std::size_t col_index, const Float arg)
  {
    static_assert(RangeDim==1,"Polynomial<PolyOrder,RangeDim,Float>::SetTerm(double) called when RangeDim != 1");
    coeffs_(col_index) = arg;
  }

  Mat Coeffs() const { return coeffs_; }

  RVec operator()(const Float& arg) const
  {
    if constexpr (RangeDim == 1) {
      Float result = coeffs_(0,PolyOrder);
      for (int i = PolyOrder-1; i >= 0; i--) {
        result = (result * arg) + coeffs_(0,i);
      }
      return result;
    }
    else {
      RVec result = coeffs_.template block<RangeDim,1>(0,PolyOrder);
      for (int i = PolyOrder-1; i >= 0; i--) {
        result = (result * arg) + coeffs_.template block<RangeDim,1>(0,i);
      }
      return result;
    }
  }

  template<std::size_t DerivOrder>
  Polynomial<PolyOrder-DerivOrder,RangeDim,Float>
  ReducedDerivative() const
  {
    Polynomial<PolyOrder-DerivOrder,RangeDim,Float> result;
    std::size_t col_index = 0;
    for (std::size_t i = DerivOrder; i <= PolyOrder; i++) {
      result.SetTerm(col_index, MonomialVectorDerivative<Float,RangeDim>(coeffs_.col(i), i, DerivOrder));
      col_index++;
    }
    return result;
  }

  //TODO validate this works by using ReducedDerivative to compare
  RVec Derivative(const std::size_t& derivative, const Float& arg) const
  {
    Float power = 1.0;
    RVec result = RVec::Zero();
    if (derivative > PolyOrder) return result;

    for (std::size_t i = derivative; i <= PolyOrder; i++) {
      result += MonomialVectorDerivative<Float,RangeDim>(coeffs_.col(i), i, derivative) * power;
      power *= arg;
    }
    return result;
  }

  template<std::size_t DerivOrder>
  RVec Derivative(const Float& arg) const
  {
    Float power = 1.0;
    RVec result = RVec::Zero();
    if constexpr (DerivOrder > PolyOrder) return result;

    for (std::size_t i = DerivOrder; i <= PolyOrder; i++) {
      result += MonomialVectorDerivative<Float,RangeDim>(coeffs_.col(i), i, DerivOrder) * power;
      power *= arg;
    }
    return result;
  }

  // OPERATORS
  Polynomial<PolyOrder, RangeDim, Float>& operator+=(Polynomial<PolyOrder, RangeDim, Float>& poly)
  {
    this->coeffs_ += poly.coeffs_;
    return *this;
  }

  Polynomial<PolyOrder, RangeDim, Float> operator+(Polynomial<PolyOrder, RangeDim, Float>& poly) const
  {
    Polynomial<PolyOrder, RangeDim, Float> result;
    result.coeffs_ = this->coeffs_ + poly.coeffs_;
    return result;
  }

};

#endif
