cmake_minimum_required(VERSION 3.8)
project(ML_Algorithms_from_scratch)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp cpp/logistic_reg/logistic_regression.cpp cpp/logistic_reg/logistic_regression.h cpp/neural_network/TraingData.cpp cpp/neural_network/TraingData.h cpp/neural_network/Neuron.cpp cpp/neural_network/Neuron.h cpp/neural_network/Net.cpp cpp/neural_network/Net.h)
add_executable(ML_Algorithms_from_scratch ${SOURCE_FILES})