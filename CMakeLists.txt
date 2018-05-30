cmake_minimum_required(VERSION 3.8)
project(ML_Algorithms_from_scratch)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp logistic_reg/logistic_regression.cpp logistic_reg/logistic_regression.h)
add_executable(ML_Algorithms_from_scratch ${SOURCE_FILES})