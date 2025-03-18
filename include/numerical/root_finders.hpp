#ifndef NUMERICAL_INCLUDE_ROOT_FINDERS_HPP
#define NUMERICAL_INCLUDE_ROOT_FINDERS_HPP


template<typename Float, typename Function1, typename Function2>
Float NewtonRootFinder(Function1& func,
                       Function2& func_derivative,
                       const Float& init_point,
                       const Float& delta_threshold = static_cast<Float>(1.0e-5))
{
  Float x = init_point;
  Float del_x;
  do {
    Float eval = func(x);
    if (eval == static_cast<Float>(0)) return x;
    del_x = eval / func_derivative(x);
    x -= del_x;
  } while((del_x > 0 ? del_x : -del_x) > delta_threshold);
  return x;
}

// template<typename Float, typename Function>
// Float SecantRootFinder(Function& func,
//                        const Float& x0,
//                        const Float& x1,
//                        const Float& delta_threshold = static_cast<Float>(1.0e-5))
// {
// 
// }

#endif
