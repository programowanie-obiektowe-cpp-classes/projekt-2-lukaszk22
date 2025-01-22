cmake_minimum_required(VERSION 3.14)
project(projekt2)

file(GLOB_RECURSE SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
add_executable(projekt2 ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Dodanie biblioteki Eigen
find_package( Eigen3 3.4 REQUIRED ) 
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen )