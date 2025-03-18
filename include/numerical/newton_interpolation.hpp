#ifndef NUMERICAL_NEWTON_INTERPOLATION_HPP
#define NUMERICAL_NEWTON_INTERPOLATION_HPP

#include <string>
#include <sstream>
#include <type_traits>
#include <array>
#include <vector>
#include <cassert>

#include <Eigen/Dense>


template<typename Float, int Dim, int ElementDim=1>
class NewtonDifferenceTable
{
private:
  typedef std::conditional_t<ElementDim==1, Float, Eigen::Vector<Float,ElementDim>> T; 
  std::array<std::array<T,Dim>,Dim> table_;

public:
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

  void Allocate(const int n)
  {
    table_.resize(n);
    for (int i = 0; i < n; i++) {
      table_[i].resize(i+1);
    }
  }

public:
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

/*
template<std::size_t PolyOrder, typename Float = double>
class NewtonPolynomial
{
public:
  typedef Eigen::Vector<Float,PolyOrder+1> Vec;

protected:
  Vec x_;
  Vec coeffs_; // First index is power of 0, second is power 1, etc. One row for each input.

public:
  NewtonPolynomial() {}
  NewtonPolynomial(const Vec& x, const Vec& y)
  {
    ConstructDifferences(x,y);
  }

  void ConstructDifferences(const Vec& x, const Vec& y)
  {
    x_ = x;
    coeffs_(0) = y(0);
    
    if constexpr (PolyOrder > 0) {
      Eigen::Array<Float,PolyOrder,PolyOrder> table;
      for (int i = 0; i < PolyOrder; i++) {
        table(i,0) = (y(i+1) - y(i)) / (x(i+1) - x(i));
      }
      coeffs_(1) = table(0,0);
      if constexpr (PolyOrder > 1) {
        for (int j = 1; j < PolyOrder; j++) {
          for (int i = j; i < PolyOrder; i++) {
            table(i,j) = ( table(i,j-1) - table(i-1,j-1) ) / ( x(i+1) - x(i-j) );
          }
          coeffs_(j+1) = table(j,j);
        }
      }
    }
  }
  
  Float operator()(const Float& arg) const
  {
    Float result = coeffs_(PolyOrder);
    for (int i = PolyOrder-1; i >= 0; i--) {
      result = result * (arg - x_(i)) + coeffs_(i);
    }
    return result;
  }
};
*/


template<int PolyOrder, int RangeDim = 1, typename Float = double>
class NewtonPolynomial
{
public:
  typedef Eigen::Vector<Float,PolyOrder+1> PVec;
  typedef typename
    std::conditional_t<RangeDim==1, Float, Eigen::Vector<Float,RangeDim>> RVec;

protected:
  std::array<Float,PolyOrder+1> x_;
  std::array<RVec,PolyOrder+1> coeffs_; // First index is power of 0, second is power 1, etc. One row for each input.

public:
  NewtonPolynomial() {}
  NewtonPolynomial(const std::array<Float,PolyOrder+1>& x,
                   const std::array<RVec,PolyOrder+1>& y)
  {
    ConstructDifferences(x,y);
  }

  NewtonPolynomial(const std::vector<Float>& x, const std::vector<RVec>& y)
  {
    ConstructDifferences(x,y);
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
  
  Float operator()(const Float& arg) const
  {
    RVec result = coeffs_[PolyOrder];
    for (int i = PolyOrder-1; i >= 0; i--) {
      result = result * (arg - x_[i]) + coeffs_[i];
    }
    return result;
  }
};


#endif
