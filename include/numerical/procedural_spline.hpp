#ifndef NUMERICAL_PROCEDURAL_SPLINE_HPP
#define NUMERICAL_PROCEDURAL_SPLINE_HPP

#include <type_traits>
#include <array>

#include <Eigen/Dense>

#include "newton_interpolation.hpp"


template<int PolyOrder, typename Functional, int RangeDim = 1, typename Scalar = double>
class ProceduralSpline
{
public:
  typedef std::conditional_t<RangeDim==1, Scalar, Eigen::Vector<Scalar,RangeDim>> RVec; 

private:
  Functional* func_;
  std::array<Scalar,PolyOrder+1> x_;
  std::array<RVec,PolyOrder+1> y_;
  NewtonDifferenceTable<Scalar,PolyOrder+1,RangeDim> table_;

  Scalar del_x_;
  bool poly_initialized_ {false};
  
  RVec Interpolate(const Scalar& arg) const
  {
    RVec result = table_(PolyOrder,PolyOrder);
    for (int i = PolyOrder-1; i >= 0; i--) {
      result = result * (arg - x_[i]) + table_(i,i);
    }
    return result;
  }

public:
  ProceduralSpline(Functional* func) : func_{func}
  {}

  ProceduralSpline(Functional* func, const Scalar init_x, const Scalar del_x)
    : func_{func}
  {
    this->Initialize(init_x, del_x);
  }

  void Initialize(const Scalar init_x, const Scalar del_x)
  {
    del_x_ = del_x;
    for (int i = 0; i <= PolyOrder; i++) {
      x_[i] = init_x + (static_cast<Scalar>(i) * del_x);
      y_[i] = func_->operator()(x_[i]);
    }
    table_.Construct(x_,y_);
    poly_initialized_ = true;
  }

  RVec Evaluate(const Scalar& arg)
  {
    if ((arg >= x_[0]) && (arg <= x_[PolyOrder])) {
      return this->Interpolate(arg);
    }
    else if ((arg > x_[PolyOrder]) &&
             (arg <= (x_[PolyOrder] + static_cast<Scalar>(PolyOrder) * del_x_))) {
      x_[0] = x_[PolyOrder];
      y_[0] = y_[PolyOrder];
      for (int i = 1; i <= PolyOrder; i++) {
        x_[i] = x_[0] + (static_cast<Scalar>(i) * del_x_);
        y_[i] = func_->operator()(x_[i]);
      }
      table_.Construct(x_,y_);
      return this->Interpolate(arg);
    }
    else if ((arg < x_[0]) &&
             (arg >= (x_[0] - static_cast<Scalar>(PolyOrder) * del_x_))) {
      x_[PolyOrder] = x_[0];
      y_[PolyOrder] = y_[0];
      for (int i = PolyOrder - 1; i >= 0; i--) {
        x_[i] = x_[PolyOrder] - (static_cast<Scalar>(i) * del_x_);
        y_[i] = func_->operator()(x_[i]);
      }
      table_.Construct(x_,y_);
      return this->Interpolate(arg);
    }
    else {
      x_[0] = arg;
      y_[0] = func_->operator()(x_[0]);
      for (int i = 1; i < PolyOrder; i++) {
        x_[i] = x_[0] + (static_cast<Scalar>(i) * del_x_);
        y_[i] = func_->operator()(x_[i]);
      }
      table_.Construct(x_,y_);
      return this->Interpolate(arg);
    }
  }
  
  RVec operator()(const Scalar& arg)
  {
    return this->Evaluate(arg);
  }
};

#endif
