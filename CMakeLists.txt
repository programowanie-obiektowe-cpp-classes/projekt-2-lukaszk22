cmake_minimum_required(VERSION 3.14)
project(projekt2)

file(GLOB_RECURSE SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
add_executable(projekt2 ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Dodanie katalogu Eigen (uwaga na ścieżkę!)
set(EIGEN3_INCLUDE_DIR "C:/eigen-3.4.0") 
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})