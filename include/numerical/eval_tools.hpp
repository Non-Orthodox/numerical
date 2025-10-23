#ifndef NUMERICAL_INCLUDE_EVAL_TOOLS_HPP
#define NUMERICAL_INCLUDE_EVAL_TOOLS_HPP

#include <tuple>

#include <Eigen/Dense>

/*
RunFactorial:
  Purpose: Given a function with a set of inputs and a vector for each input, this will run the function for every combination of the different inputs specified in the vectors
  Overloads: One version uses std::vector for the input ranges, the other uses dynamic Eigen vectors. This may be generalized to generic eigen dense types in the future.
  Basic example: If a function "f" has two integer inputs "a" and "b", then the vector {1,2,3} for a and {1,2,3} for b would run the function 9 times for each combination of these inputs
*/

template<typename Func, typename TupleType, typename InType>
void RunFactorial_impl(Func func, TupleType& tup, std::vector<InType>& in)
{
  for (std::size_t i = 0; i < in.size(); i++) {
    std::get<std::tuple_size_v<TupleType> - 1>(tup) = in[i];
    std::apply(func, tup);
  }
}

template<typename Func, typename TupleType, typename InType, typename... InTypes>
void RunFactorial_impl(Func func, TupleType& tup, std::vector<InType>& in1, std::vector<InTypes>&... ins)
{
  for (std::size_t i = 0; i < in1.size(); i++) {
  std::get<std::tuple_size_v<TupleType> - sizeof...(InTypes) - 1>(tup) = in1[i];
    RunFactorial_impl(func, tup, ins...);
  }
}

template<typename OutType, typename... InTypes>
void RunFactorial(std::function<OutType(InTypes...)>& func, std::vector<InTypes>&... in_vectors)
{
  std::tuple<InTypes...> tup;
  RunFactorial_impl(func, tup, in_vectors...);
}


template<typename Func, typename TupleType, typename InType>
void RunFactorial_impl(Func func, TupleType& tup, Eigen::Vector<InType,Eigen::Dynamic>& in)
{
  for (std::size_t i = 0; i < in.size(); i++) {
    std::get<std::tuple_size_v<TupleType> - 1>(tup) = in(i);
    std::apply(func, tup);
  }
}

template<typename Func, typename TupleType, typename InType, typename... InTypes>
void RunFactorial_impl(Func func, TupleType& tup, Eigen::Vector<InType,Eigen::Dynamic>& in1, Eigen::Vector<InTypes,Eigen::Dynamic>&... ins)
{
  for (std::size_t i = 0; i < in1.size(); i++) {
  std::get<std::tuple_size_v<TupleType> - sizeof...(InTypes) - 1>(tup) = in1(i);
    RunFactorial_impl(func, tup, ins...);
  }
}

template<typename OutType, typename... InTypes>
void RunFactorial(std::function<OutType(InTypes...)>& func, Eigen::Vector<InTypes,Eigen::Dynamic>&... in_vectors)
{
  std::tuple<InTypes...> tup;
  RunFactorial_impl(func, tup, in_vectors...);
}


#endif
