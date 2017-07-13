# CMake entry point
cmake_minimum_required (VERSION 2.6)
project (Hexmeshing)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(
    /usr/include/
    /usr/include/eigen3
    /usr/local/include/
    /usr/local/include/vtk-6.3/
	./src
    .
)

set(ALL_LIBS
)

add_definitions(
#	-DTW_STATIC
#	-DTW_NO_LIB_PRAGMA
#	-DTW_NO_DIRECT3D
#	-DGLEW_STATIC
#	-D_CRT_SECURE_NO_WARNINGS
)

set(SRC
    src/Metric.h
    src/Metric.cpp
    src/Distance.h
    src/Distance.cpp
    src/Initialization.h
    src/Initialization.cpp
    src/IOHandler.h
    src/IOHandler.cpp
	src/PCA_Cluster.h
	src/PCA_Cluster.cpp
	src/main.cpp
)

add_executable(cluster ${SRC} src/main.cpp)
target_link_libraries(cluster ${ALL_LIBS})
