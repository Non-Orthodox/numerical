#include <random>
#include <iostream>
#include <array>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <Eigen/Dense>

#include <numerical/newton_interpolation.hpp>

/*
make sure to test all equivalence classes
*/



TEMPLATE_TEST_CASE("Difference table with Dim of 1", "", float, double, long double)
{
  TestType x = static_cast<TestType>(0);
  TestType y = static_cast<TestType>(1);
  
  SECTION("static size") {
    NewtonDifferenceTable<TestType,1,1> table1(&x,&y,1);
    REQUIRE( table1(0,0) == static_cast<TestType>(1) );
    
    NewtonDifferenceTable<TestType,1,1> table2;
    table2.Construct(&x,&y,1);
    REQUIRE( table2(0,0) == static_cast<TestType>(1) );
  }
  SECTION("dynamic size") {
    NewtonDifferenceTable<TestType,-1,1> table1(&x,&y,1);
    REQUIRE( table1(0,0) == static_cast<TestType>(1) );
    
    NewtonDifferenceTable<TestType,-1,1> table2;
    table2.Construct(&x,&y,1);
    REQUIRE( table2(0,0) == static_cast<TestType>(1) );
  }
}

template<typename Float, int TableDim, int VecDim>
void TestStaticTableConsistency()
{
  static_assert(TableDim > 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Float> dist(0.0, 1.0);

  typedef typename NewtonDifferenceTable<Float,TableDim,VecDim>::DataType DType;
  for (int it = 0; it < 100; it++) {
    NewtonDifferenceTable<Float,TableDim,VecDim> table;
    std::array<Float,TableDim> x;
    std::array<DType,TableDim> y;
    x[0] = (dist(gen) * static_cast<Float>(100));
    for (int i = 1; i < TableDim; i++) {
      x[i] = x[i-1] + (dist(gen) * static_cast<Float>(10)) + static_cast<Float>(1.0e-3);
    }
    for (int i = 0; i < TableDim; i++) {
      if constexpr (VecDim != 1) {
        for (int j = 0; j < VecDim; j++) {
          y[i](j) = (dist(gen) * static_cast<Float>(2.0e8)) - static_cast<Float>(1.0e8);
        }
      }
      else {
        y[i] = (dist(gen) * static_cast<Float>(2.0e8)) - static_cast<Float>(1.0e8);
      }
    }
    table.Construct(x,y);
    
    bool correct_construction = true;
    for (int i = 0; i < TableDim; i++) {
      correct_construction &= (table(i,0) == y[i]);
    }
    for (int j = 1; j < TableDim; j++) {
      for (int i = j; i < TableDim; i++) {
        DType correct_value = (table(i,j-1) - table(i-1,j-1)) / (x[i] - x[i-j]);
        correct_construction &= (table(i,j) == correct_value);
      }
    }
    REQUIRE(correct_construction);
  }
}

template<typename Float, int VecDim>
void TestStaticTableConsistency(int table_dim)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Float> dist(0.0, 1.0);

  typedef typename NewtonDifferenceTable<Float,-1,VecDim>::DataType DType;
  for (int it = 0; it < 100; it++) {
    NewtonDifferenceTable<Float,-1,VecDim> table;
    std::vector<Float> x(table_dim);
    std::vector<DType> y(table_dim);
    x[0] = (dist(gen) * static_cast<Float>(100));
    for (int i = 1; i < table_dim; i++) {
      x[i] = x[i-1] + (dist(gen) * static_cast<Float>(10)) + static_cast<Float>(1.0e-3);
    }
    for (int i = 0; i < table_dim; i++) {
      if constexpr (VecDim != 1) {
        for (int j = 0; j < VecDim; j++) {
          y[i](j) = (dist(gen) * static_cast<Float>(2.0e2)) - static_cast<Float>(1.0e2);
        }
      }
      else {
        y[i] = (dist(gen) * static_cast<Float>(2.0e2)) - static_cast<Float>(1.0e2);
      }
    }
    table.Construct(x,y);
    
    bool correct_construction = true;
    for (int i = 0; i < table_dim; i++) {
      correct_construction &= (table(i,0) == y[i]);
    }
    for (int j = 1; j < table_dim; j++) {
      for (int i = j; i < table_dim; i++) {
        DType correct_value = (table(i,j-1) - table(i-1,j-1)) / (x[i] - x[i-j]);
        correct_construction &= (table(i,j) == correct_value);
      }
    }
    REQUIRE(correct_construction);
  }
}

// in case the code gets refactored, make sure the relationships between elements is the same
TEMPLATE_TEST_CASE("Table relational consistency", "", float, double, long double)
{
  SECTION("static size") {
    TestStaticTableConsistency<TestType,2,1>();
    TestStaticTableConsistency<TestType,3,1>();
    TestStaticTableConsistency<TestType,4,1>();
    TestStaticTableConsistency<TestType,2,2>();
    TestStaticTableConsistency<TestType,3,2>();
    TestStaticTableConsistency<TestType,4,2>();
    TestStaticTableConsistency<TestType,2,3>();
    TestStaticTableConsistency<TestType,2,3>();
    TestStaticTableConsistency<TestType,3,3>();
    TestStaticTableConsistency<TestType,4,4>();
    TestStaticTableConsistency<TestType,3,4>();
    TestStaticTableConsistency<TestType,4,4>();
  }
  SECTION("dynamic size") {
    TestStaticTableConsistency<TestType,1>(2);
    TestStaticTableConsistency<TestType,1>(3);
    TestStaticTableConsistency<TestType,1>(4);
    TestStaticTableConsistency<TestType,2>(2);
    TestStaticTableConsistency<TestType,2>(3);
    TestStaticTableConsistency<TestType,2>(4);
    TestStaticTableConsistency<TestType,3>(2);
    TestStaticTableConsistency<TestType,3>(3);
    TestStaticTableConsistency<TestType,3>(4);
    TestStaticTableConsistency<TestType,4>(2);
    TestStaticTableConsistency<TestType,4>(3);
    TestStaticTableConsistency<TestType,4>(4);
  }
}

template<int Order, int RDim, typename Float>
void NewtonControlPointTest()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Float> dist(0.0, 1.0);

  typedef typename NewtonPolynomial<Order,RDim,Float>::RVec RVec;
  for (int it = 0; it < 100; it++) {
    std::array<Float,Order+1> x;
    std::array<RVec,Order+1> y;
    x[0] = (dist(gen) * static_cast<Float>(100));
    for (int i = 1; i <= Order; i++) {
      x[i] = x[i-1] + (dist(gen) * static_cast<Float>(10)) + static_cast<Float>(1.0e-3);
    }
    for (int i = 0; i <= Order; i++) {
      if constexpr (RDim != 1) {
        for (int j = 0; j < RDim; j++) {
          y[i](j) = (dist(gen) * static_cast<Float>(2.0e8)) - static_cast<Float>(1.0e8);
        }
      }
      else {
        y[i] = (dist(gen) * static_cast<Float>(2.0e8)) - static_cast<Float>(1.0e8);
      }
    }

    NewtonPolynomial<Order,RDim,Float> poly(x,y);
    for (int i = 0; i <= Order; i++) {
      if constexpr (RDim == 1) {
        REQUIRE( std::abs((poly(x[i]) / y[i]) - Float(1)) <= Float(1.0e-4) );
      }
      else {
        for (int j = 0; j < RDim; j++) {
          REQUIRE( std::abs((poly(x[i])(j) / y[i](j)) - Float(1)) <= Float(1.0e-4) );
        }
      }
    }
  }
}

// not doing float since the precision degrades pretty rapidly with polynomial order compared to double
TEMPLATE_TEST_CASE("Test static-order newton polynomial control point relations", "", double, long double)
{
  NewtonControlPointTest<1,1,TestType>();
  NewtonControlPointTest<2,1,TestType>();
  NewtonControlPointTest<4,1,TestType>();
  NewtonControlPointTest<8,1,TestType>();
  NewtonControlPointTest<1,2,TestType>();
  NewtonControlPointTest<2,2,TestType>();
  NewtonControlPointTest<4,2,TestType>();
  NewtonControlPointTest<8,2,TestType>();
  NewtonControlPointTest<1,4,TestType>();
  NewtonControlPointTest<2,4,TestType>();
  NewtonControlPointTest<4,4,TestType>();
  NewtonControlPointTest<8,4,TestType>();
}

template<typename T,int Size>
T PolyEval(const std::array<T,Size>& c, const T& x)
{
  T result = c[0];
  T pow = T(1);
  for (int i = 1; i < Size; i++) {
    pow *= x; // now pow is x^i
    result += c[i] * pow;
  }
  return result;
}

TEST_CASE("Test Newton polynomial derivative generation")
{
  // polynomial 1 + 5x + 2x^2 - 3x^3
  std::array<double,4> c1;
  c1[0] = 1.0;
  c1[1] = 5.0;
  c1[2] = 2.0;
  c1[3] = -3.0;

  // derivative 5 + 4x - 9x^2
  std::array<double,3> c2;
  c2[0] = 5.0;
  c2[1] = 4.0;
  c2[2] = -9.0;
  
  std::array<double,2> c3;
  c3[0] = 4.0;
  c3[1] = -18.0;

  double c4 = -18.0;

  std::array<double,4> x, y;
  for (int i = 0; i < 4; i++) {
    x[i] = static_cast<double>(i);
    y[i] = PolyEval<double,4>(c1, x[i]);
  }

  NewtonPolynomial<3,1,double> poly(x,y);
  double eval, eval_prime;
  NewtonPolynomial<2,1,double> d = poly.Derivative<1>();
  NewtonPolynomial<1,1,double> dd = poly.Derivative<2>();
  NewtonPolynomial<0,1,double> ddd = poly.Derivative<3>();
  NewtonPolynomial<0,1,double> dddd = poly.Derivative<4>();
  for (int i = 0; i < 3; i++) {
    double x = static_cast<double>(i);
    REQUIRE( d(x) == PolyEval<double,3>(c2,x) );
    REQUIRE( dd(x) == PolyEval<double,2>(c3,x) );
    REQUIRE( ddd(x) == c4 );
    REQUIRE( dddd(x) == 0.0 );
  }
}
