cmake_minimum_required(VERSION 3.16)
project(mdetector_cpu_adapter_candidate LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_build_inputs.py
                RESULT_VARIABLE GENERATE_RESULT)
if(NOT GENERATE_RESULT EQUAL 0)
  message(FATAL_ERROR "Pinned vendor or transport input generation failed")
endif()
add_executable(compat_probe native/compat_probe.cpp)
target_include_directories(compat_probe PRIVATE native generated)

# OFF by default: missing native dependencies and full-core compatibility are unresolved.
# No executable initializes or runs the detector in this candidate.
option(BUILD_PINNED_DETECTOR "Attempt the unchanged pinned author core static library" OFF)
if(BUILD_PINNED_DETECTOR)
  find_package(Eigen3 3.3.4 REQUIRED NO_MODULE)
  find_package(Boost REQUIRED)
  find_package(PCL 1.8 REQUIRED COMPONENTS common filters search kdtree)
  find_package(TBB REQUIRED)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  find_package(Threads REQUIRED)
  add_library(mdetector_candidate STATIC native/bridge.cpp
      vendor/src/DynObjFilter.cpp vendor/src/DynObjCluster.cpp)
  target_include_directories(mdetector_candidate BEFORE PRIVATE
      generated/transport_include native generated vendor/include
      ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
  target_compile_options(mdetector_candidate PRIVATE -include
      ${CMAKE_CURRENT_SOURCE_DIR}/native/prelude.hpp ${PCL_DEFINITIONS})
  target_link_libraries(mdetector_candidate PRIVATE Eigen3::Eigen
      ${PCL_LIBRARIES} TBB::tbb OpenMP::OpenMP_CXX Threads::Threads)
endif()
