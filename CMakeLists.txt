cmake_minimum_required (VERSION 2.6)
project (arma_davidson)

add_definitions(-DDEBUG -p -g)
#add_definitions(-O3 -g -c)

FIND_LIBRARY(ARMADILLO armadillo)
MESSAGE(STATUS "Armadillo Library location: " ${ARMADILLO})

add_library( arma_davidson 
	SHARED
	Davidson.cpp 
	Davidson.h 
	)

add_executable( test1 test1.cpp)

target_link_libraries( test1 arma_davidson ${ARMADILLO})


