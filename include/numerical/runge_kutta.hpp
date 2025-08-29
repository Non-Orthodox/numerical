#ifndef NUMERICAL_INCLUDE_RUNGE_KUTTA_HPP
#define NUMERICAL_INCLUDE_RUNGE_KUTTA_HPP

#include <functional>
#include <cassert>

#include <Eigen/Dense>


template<typename Float, int S>
struct ButcherTable
{
  Eigen::Array<Float,S,S> a;
  Eigen::Vector<Float,S> b;
  Eigen::Vector<Float,S> c;
};


template<typename T>
void Rk4Table(ButcherTable<T,4>& table)
{
  table.a = {{0., 0., 0., 0.},
             {0.5, 0., 0., 0.},
             {0., 0.5, 0., 0.},
             {0., 0., 1., 0.}};

  table.b(0) = (1.0 / 6.0);
  table.b(1) = (1.0 / 3.0);
  table.b(2) = table.b(1);
  table.b(3) = table.b(0);
  
  table.c(0) = 0.0;
  table.c(1) = 0.5;
  table.c(2) = 0.5;
  table.c(3) = 1.0;
}


template<int StateDim, typename Float = double>
Vector<Float,StateDim> RK4(Vector<Float,StateDim> x, 
  std::function<Vector<Float,StateDim>(Float, Vector<Float,StateDim>)> f,
  Float t, // time stamp of x
  Float h)
{
  typedef Vector<Float,StateDim> Vec;
  Vec a = f(t, x);
  Vec b = f(t + (h/2.0), x + ((h/2.0)*a));
  Vec c = f(t + (h/2.0), x + ((h/2.0)*b));
  Vec d = f(t + h, x + (h*c));
  return ( x + ((h/6.0) * (a + (2.0*b) + (2.0*c) + d)) );
}


template<int StateDim, typename Float = double>
void RK45(
  Vector<Float,StateDim>& x, 
  const std::function<Vector<Float,StateDim>(Float,Vector<Float,StateDim>)> f,
  Float& t,
  Float& h,
  Float& h_next,
  Float epsilon
  )
{
  assert(epsilon > 0);
  assert(h > 0);
  assert(h_next > 0);

  Vector<Float,StateDim> approx_error, k1, k2, k3, k4, k5, k6;
  do {
    k1 = h * f(t,x);
    k2 = h * f(t + (h / 4.0), 
      x + (k1/4.0));
    k3 = h * f(t + (3.0*h / 8.0),
      x + (3.0*k1/32.0) + (9.0*k2/32.0));
    k4 = h * f(t + (12.0*h / 13.0),
      x + (1932.0*k1/2197.0) - (7200.0*k2/2197.0) + (7296.0*k3/2197.0));
    k5 = h * f(t + h,
      x + (439.0*k1/216.0) - (8.0*k2) + (3680.0*k3/513.0) - (845.0*k4/4104.0));
    k6 = h * f(t + (h / 2.0),
      x - (8.0*k1/27.0) + (2.0*k2) - (3544.0*k3/2565.0) + (1859.0*k4/4104.0) - (11.0*k5/40.0));
    
    approx_error = -(k1/360.0) + (128.0*k3/4275.0) + (2187.0*k4/75240.0) - (k5/50.0) - (2.0*k6/55.0);
    approx_error = (approx_error >= 0) ? approx_error : -approx_error;

    h_next = 0.9 * h * std::pow(epsilon / approx_error, 0.2);
    if (approx_error > epsilon) {
      h = h_next;
    }
  }
  while(approx_error > epsilon);

  x += (16.0*k1/135.0) + (6656.0*k3/12825.0) + (28561.0*k4/56430.0) - (9.0*k5/50.0) + (2.0*k6/55.0);
}


template<typename Float>
constexpr std::array<Float,11> dexp_inv_coeffs = {
  1.,             // 0
  (-1./2.),       // 1
  (1./12.),       // 2
  0,
  (-1./720.),     // 4
  0,
  (1./30240.),    // 6
  0,
  (-1./1209600),  // 8
  0,
  (1./47900160)   // 10
};


template<class Derived1, class Derived2>
Eigen::MatrixBase<Derived1> dexp_inv(
  const Eigen::MatrixBase<Derived2>& u, // member of lie algebra
  const Eigen::MatrixBase<Derived1>& v,  // member of lie group
  std::size_t q = 10)
{
  Eigen::MatrixBase<Derived1> ad = v;
  Eigen::MatrixBase<Derived1> result = dexp_inv_coeffs<typename Derived1::Scalar>[0] * ad;
  for (std::size_t k = 1; k <= q; k++) {
    ad = ((u * ad) - (ad * u));
    result += dexp_inv_coeffs<typename Derived1::Scalar>[k] * ad;
  }
}


template<class Derived>
Eigen::MatrixBase<Derived>
expm(Eigen::MatrixBase<Derived>& mat, unsigned int it = 5)
{
  assert(mat.rows() == mat.cols());
  typedef typename Derived::Scalar T;
  constexpr int dim = (Derived::RowsAtCompileTime > Derived::ColsAtCompileTime) ? 
                      Derived::RowsAtCompileTime : Derived::ColsAtCompileTime;
  typedef Eigen::Matrix<T,dim,dim> Mat;
  Mat power = Mat::Identity(mat.rows());
  T factor = 1.0;
  Mat result = power;
  for (int i = 1; i <= it; i++) {
    factor /= static_cast<T>(i);
    power *= mat;
    result += factor * power;
  }
  return result;
}


template<class Derived, typename Float>
Eigen::MatrixBase<Derived> RKMK4(
  const std::function<Eigen::MatrixBase<Derived>(Eigen::MatrixBase<Derived>,Float)>& f, 
  const Eigen::MatrixBase<Derived>& y,
  Float t,
  Float h
  )
{
  typedef Eigen::MatrixBase<Derived> Mat;

  ButcherTable<Float,4> table;
  Rk4Table(table);
  std::array<Mat,4> ks;
  for (std::size_t i = 0; i < 4; i++) {
    Mat u = Mat::Zero();
    for (int j = 0; j < i; j++) {
      u += table.a(i,j) * ks[j];
    }
    u *= h;
    ks[i] = f(expm(u) * y, t + h * table.c(i));
    ks[i] = dexp_inv(u,ks[i]);
  }
  Mat v = Mat::Zero();
  for (int j = 0; j < 4; j++) {
    v += table.b(j) * ks[j];
  }
  return expm(h * v) * y.derived();
}

#endif
