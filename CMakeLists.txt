cmake_minimum_required(VERSION 3.12)
project(NUMERICAL_TESTS)
set(CMAKE_CXX_STANDARD 20)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(newton_interp_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/newton_interp_tests.cpp)
target_include_directories(newton_interp_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(newton_interp_tests PUBLIC Eigen3::Eigen)

