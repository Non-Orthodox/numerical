#ifndef NUMERICAL_NEWTON_INTERPOLATION_HPP
#define NUMERICAL_NEWTON_INTERPOLATION_HPP

#include <string>
#include <sstream>
#include <type_traits>
#include <array>
#include <vector>
#include <type_traits>
#include <cassert>

#include <Eigen/Dense>

//TODO maybe make a general Hermitian difference table


/* TODO could make LowerTriangleArray
  -make one-dimensional array, map (i,j) to it with the fact that sum[1,n] = n(n+1)/2
*/


// Dim is the rows and columns of the table, ElementDim is the function output dimension
template<typename Float, int Dim, int ElementDim=1>
class NewtonDifferenceTable
{
private:
  typedef std::conditional_t<ElementDim==1, Float, Eigen::Vector<Float,ElementDim>> T; 
  std::array<std::array<T,Dim>,Dim> table_; //TODO could make this more memory efficient since only triangle used
  static_assert((Dim > 0) || (Dim == -1), "NewtonDifferenceTable cannot have negative size");
  static_assert(ElementDim > 0, "NewtonDifferenceTable must consist of elements with positive dimension");

public:
  typedef T DataType;

  NewtonDifferenceTable() {}

  NewtonDifferenceTable(const Float* x, const T* y, const std::size_t size)
  {
    this->Construct(x,y,size);
  }

  NewtonDifferenceTable(const std::array<Float,Dim>& x, const std::array<T,Dim>& y)
  {
    this->Construct(x,y);
  }

  NewtonDifferenceTable(const std::vector<Float>& x, const std::vector<T>& y)
  {
    this->Construct(x,y);
  }

  T& operator()(const int i, const int j)
  {
    return table_[i][j];
  }
  
  T operator()(const int i, const int j) const
  {
    return table_[i][j];
  }

  std::array<std::array<T,Dim>,Dim> Table() const
  { return table_; }

  void Construct(const Float* x, const T* y, const std::size_t size)
  {
    assert(size >= Dim);
    // this->Allocate(size);
    
    for (int i = 0; i < Dim; i++) {
      table_[i][0] = y[i];
    }
    if constexpr (Dim > 1) {
      for (int j = 1; j < Dim; j++) {
        for (int i = j; i < Dim; i++) {
          table_[i][j] = ( table_[i][j-1] - table_[i-1][j-1] ) / ( x[i] - x[i-j] );
        }
      }
    }
  }

  void Construct(const std::array<Float,Dim>& x,
                            const std::array<T,Dim>& y)
  {
    this->Construct(x.data(), y.data(), Dim);
  }

  void Construct(const std::vector<Float>& x, const std::vector<T>& y)
  {
    std::size_t size = (x.size() < y.size()) ? x.size() : y.size();
    this->Construct(x.data(), y.data(), size);
  }

  std::string str() const
  {
    std::stringstream ss;
    for (int i = 0; i < Dim; i++) {
      for (int j = 0; j <= i; j++) {
        ss << table_[i][j] << ' ';
      }
      ss << '\n';
    }
    return ss.str();
  }

};


template<typename Float, int ElementDim>
class NewtonDifferenceTable<Float,-1,ElementDim>
{
private:
  typedef std::conditional_t<ElementDim==1, Float, Eigen::Vector<Float,ElementDim>> T;
  std::vector<std::vector<T>> table_;
  static_assert(ElementDim > 0, "NewtonDifferenceTable must consist of elements with positive dimension");

  void Allocate(const int n)
  {
    table_.resize(n);
    for (int i = 0; i < n; i++) {
      table_[i].resize(i+1);
    }
  }

public:
  typedef T DataType;

  NewtonDifferenceTable() {}

  NewtonDifferenceTable(const Float* x, const T* y, const std::size_t size)
  {
    this->Construct(x,y,size);
  }

  template<std::size_t Dim>
  NewtonDifferenceTable(const std::array<Float,Dim>& x, const std::array<T,Dim>& y)
  {
    this->Construct(x,y);
  }

  NewtonDifferenceTable(const std::vector<Float>& x, const std::vector<T>& y)
  {
    this->Construct(x,y);
  }

  T& operator()(const int i, const int j)
  {
    return table_[i][j];
  }
  
  T operator()(const int i, const int j) const
  {
    return table_[i][j];
  }

  std::vector<std::vector<T>> Table() const
  { return table_; }

  void Construct(const Float* x, const T* y, const std::size_t size)
  {
    this->Allocate(size);
    
    for (int i = 0; i < size; i++) {
      table_[i][0] = y[i];
    }
    for (int j = 1; j < size; j++) {
      for (int i = j; i < size; i++) {
        table_[i][j] = ( table_[i][j-1] - table_[i-1][j-1] ) / ( x[i] - x[i-j] );
      }
    }
  }

  template<std::size_t Dim>
  void Construct(const std::array<Float,Dim>& x, const std::array<T,Dim>& y)
  {
    this->Construct(x.data(), y.data(), Dim);
  }

  void Construct(const std::vector<Float>& x, const std::vector<T>& y)
  {
    std::size_t size = (x.size() < y.size()) ? x.size() : y.size();
    this->Construct(x.data(), y.data(), size);
  }

  std::string str() const
  {
    std::stringstream ss;
    for (int i = 0; i < table_.size(); i++) {
      for (int j = 0; j <= i; j++) {
        ss << table_[i][j] << ' ';
      }
      ss << '\n';
    }
    return ss.str();
  }
};


//TODO specialization with dynamic poly order
template<int PolyOrder, int RangeDim = 1, typename Float = double>
class NewtonPolynomial
{
public:
  static_assert(PolyOrder >= 0, "NewtonPolynomial order must be positive");
  static_assert(RangeDim > 0, "NewtonPolynomial output dimension (RangeDim) must be positive");
  typedef Eigen::Vector<Float,PolyOrder+1> PVec;
  typedef typename
    std::conditional_t<RangeDim==1, Float, Eigen::Vector<Float,RangeDim>> RVec;

  static RVec zero()
  {
    if constexpr (RangeDim==1) {
      return Float(0);
    }
    else {
      return RVec::Zero();
    }
  }

  static NewtonPolynomial<0,RangeDim,Float> ZeroPolynomial()
  {
    return NewtonPolynomial<0,RangeDim,Float>(
      std::array<Float,1>({0}),std::array<RVec,1>({zero()}));
  }

protected:
  std::array<Float,PolyOrder+1> x_;
  std::array<RVec,PolyOrder+1> coeffs_; // First index is power of 0, second is power 1, etc. One row for each input.

public:
  NewtonPolynomial() {}
  NewtonPolynomial(const std::array<Float,PolyOrder+1>& x,
                   const std::array<RVec,PolyOrder+1>& y)
  {
    this->ConstructDifferences(x,y);
  }

  NewtonPolynomial(const std::vector<Float>& x, const std::vector<RVec>& y)
  {
    this->ConstructDifferences(x,y);
  }

  Float x(const std::size_t index) const
  {
    return x_[index];
  }

  void ConstructDifferences(const std::vector<Float>& x, const std::vector<RVec>& y)
  {
    assert((x.size() > PolyOrder) && (y.size() > PolyOrder));
    NewtonDifferenceTable<Float,PolyOrder+1,RangeDim> table(x,y);
    for (int i = 0; i <= PolyOrder; i++) {
      x_[i] = x[i];
      coeffs_[i] = table(i,i);
    }
  }

  void ConstructDifferences(const std::array<Float,PolyOrder+1>& x,
                            const std::array<RVec,PolyOrder+1>& y)
  {
    x_ = x;
    NewtonDifferenceTable<Float,PolyOrder+1,RangeDim> table(x,y);
    for (int i = 0; i <= PolyOrder; i++) {
      coeffs_[i] = table(i,i);
    }
  }
  
  //TODO change to be generic eigen dense objects that have the correct size
  void Evaluate(const Float& arg, RVec& out, RVec& out_prime) const
  {
    out = coeffs_[PolyOrder];
    out_prime = this->zero();
    for (int i = PolyOrder-1; i >= 0; i--) {
      Float diff = arg - x_[i];
      out_prime = out + (diff * out_prime);
      out = (out * diff) + coeffs_[i];
    }
  }

  void Evaluate(const Float& arg, RVec& out) const
  {
    out = coeffs_[PolyOrder];
    for (int i = PolyOrder-1; i >= 0; i--) {
      out = (out * (arg - x_[i])) + coeffs_[i];
    }
  }

  RVec operator()(const Float& arg) const
  {
    RVec result;
    this->Evaluate(arg,result);
    return result;
  }

/*
  template<int _Order, int _RangeDim, typename _Float>
  bool operator==(const NewtonPolynomial<_Order,_RangeDim,_Float>& other)
  {
    if constexpr (std::is_same_v<Float,_Float>) {

    }
    else {
      return false;
    }
  }
*/

  void Derivative(NewtonPolynomial<PolyOrder-1,RangeDim,Float>& poly)
  {
    std::array<Float,PolyOrder> x;
    std::array<RVec,PolyOrder> y;
    Float temp;
    for (int i = 0; i < PolyOrder; i++) {
      x[i] = x_[i];
      this->Evaluate(x[i],temp,y[i]);
    }
    poly.ConstructDifferences(x,y);
  }

  template<int DerivOrder = 1>
  auto Derivative() const
  {
    static_assert(DerivOrder >= 0, "NewtonPolynomial::Derivative<DerivOrder> cannot have negative DerivOrder");
    if constexpr (DerivOrder == 0) {
      return *this;
    }
    else if constexpr (PolyOrder == 0) {
      return ZeroPolynomial();
    }
    else {
      std::array<Float,PolyOrder> x;
      std::array<RVec,PolyOrder> y;
      Float temp;
      //TODO could change chosen x points
      for (int i = 0; i < PolyOrder; i++) {
        x[i] = x_[i];
        this->Evaluate(x[i],temp,y[i]);
      }
      if constexpr (DerivOrder == 1) {
        return NewtonPolynomial<PolyOrder-1,RangeDim,Float>(x,y);
      }
      else {
        NewtonPolynomial<PolyOrder-1,RangeDim,Float> deriv(x,y);
        return deriv.template Derivative<DerivOrder-1>();
      }
    }
  }

  std::string str() const
  {
    std::stringstream ss;
    for (int i = 0; i <= PolyOrder; i++) {
      ss << coeffs_[i] << " ";
    }
    return ss.str();
  }
};


#endif
