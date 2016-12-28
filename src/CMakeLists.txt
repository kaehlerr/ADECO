
add_definitions(-DNO_HDF5)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

find_package(OpenMP)
if (OPENMP_FOUND)
    list (APPEND CMAKE_C_FLAGS " ${CMAKE_C_FLAGS}  ${OpenMP_C_FLAGS} ")
    list (APPEND CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}  ${OpenMP_CXX_FLAGS} ")
endif()

if (USE_GPU STREQUAL "TRUE" )
	find_package(CUDA REQUIRED)
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_37;-std=c++11;-O3;-DVERBOSE")
    	list(APPEND CUDA_NVCC_FLAGS "--maxrregcount 200; -Xptxas -v,-abi=no;-Xcompiler; -rdynamic")
    	set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-bindir g++")
    	cuda_add_library( cuda_part AdDeposit2Grid.cu )
	add_executable( dm_resampling_gpu main.cpp )
    	target_link_libraries(dm_resampling_gpu cuda_part ${MPI_LIBRARIES} pthread)
    	set_property(TARGET dm_resampling_gpu PROPERTY CXX_STANDARD 11)
else()
	add_executable( dm_resampling_cpu main.cpp AdDeposit2Grid.cpp )
	target_link_libraries( dm_resampling_cpu ${MPI_LIBRARIES} pthread )
    	set_property(TARGET dm_resampling_cpu PROPERTY CXX_STANDARD 11)
endif()

#set(CMAKE_C_COMPILER mpicc )
#set(CMAKE_CXX_COMPILER mpicxx )
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")





