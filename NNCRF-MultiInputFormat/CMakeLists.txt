project(MinervaSegs)

include_directories(
  basic
  /opt/mshadow/
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda

#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)

add_executable(DiscreteCRFMMLabeler DiscreteCRFMMLabeler.cpp)
add_executable(DenseCRFMMLabeler DenseCRFMMLabeler.cpp)
add_executable(DcombCRFMMLabeler DcombCRFMMLabeler.cpp)
add_executable(MultiDiscreteCRFMMLabeler MultiDiscreteCRFMMLabeler.cpp)
add_executable(MultiDenseCRFMMLabeler MultiDenseCRFMMLabeler.cpp)
add_executable(MultiDcombCRFMMLabeler MultiDcombCRFMMLabeler.cpp)



target_link_libraries(DiscreteCRFMMLabeler openblas)
target_link_libraries(DenseCRFMMLabeler openblas)
target_link_libraries(DcombCRFMMLabeler openblas)
target_link_libraries(MultiDiscreteCRFMMLabeler openblas)
target_link_libraries(MultiDenseCRFMMLabeler openblas)
target_link_libraries(MultiDcombCRFMMLabeler openblas)


