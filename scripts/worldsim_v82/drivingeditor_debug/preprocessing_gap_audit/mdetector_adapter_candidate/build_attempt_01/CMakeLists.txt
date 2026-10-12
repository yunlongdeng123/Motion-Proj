cmake_minimum_required(VERSION 3.16)
project(mdetector_isolated_native_compile LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(CANDIDATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(PRIVATE_PREFIX "${CANDIDATE_DIR}/deps/prefix/usr")
list(PREPEND CMAKE_PREFIX_PATH "${PRIVATE_PREFIX}")
find_package(Eigen3 3.3.4 REQUIRED NO_MODULE)
find_package(Boost 1.65 REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

# Build portability only: the distribution PCLConfig demands extra Boost binary
# components before inspecting these sources. Use the real private PCL headers
# and shared libraries directly for this compile-only static-library target.
set(PRIVATE_PCL_INCLUDE "${PRIVATE_PREFIX}/include/pcl-1.12")
if(NOT EXISTS "${PRIVATE_PCL_INCLUDE}/pcl/pcl_config.h")
  message(FATAL_ERROR "Verified private PCL headers missing")
endif()
set(PRIVATE_PCL_LIBS "")
foreach(component common filters search kdtree sample_consensus octree)
  find_library(PRIVATE_PCL_${component} NAMES pcl_${component}
    PATHS "${PRIVATE_PREFIX}/lib/x86_64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
  list(APPEND PRIVATE_PCL_LIBS "${PRIVATE_PCL_${component}}")
endforeach()
add_library(mdetector_candidate STATIC
  "${CANDIDATE_DIR}/native/bridge.cpp"
  "${CANDIDATE_DIR}/vendor/src/DynObjFilter.cpp"
  "${CANDIDATE_DIR}/vendor/src/DynObjCluster.cpp")
target_include_directories(mdetector_candidate BEFORE PRIVATE
  "${CANDIDATE_DIR}/generated/transport_include"
  "${CANDIDATE_DIR}/native" "${CANDIDATE_DIR}/generated"
  "${CANDIDATE_DIR}/vendor/include" "${PRIVATE_PREFIX}/include"
  "${PRIVATE_PCL_INCLUDE}" ${Boost_INCLUDE_DIRS})
target_compile_options(mdetector_candidate PRIVATE
  -include "${CANDIDATE_DIR}/native/prelude.hpp")
target_link_libraries(mdetector_candidate PRIVATE Eigen3::Eigen
  ${PRIVATE_PCL_LIBS} TBB::tbb OpenMP::OpenMP_CXX Threads::Threads)

# Deliberately no executable, test registration or constructor invocation.
