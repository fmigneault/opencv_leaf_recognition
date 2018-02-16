#========================================
# OpenCV's Leaf Recogntion samples (opencvLeafRecognition)
#========================================

cmake_minimum_required(VERSION 2.8)

# =========================
# project
# =========================

set(cvLeafRecog_PROJECT "cvLeafRecog")
project(${cvLeafRecog_PROJECT})

if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
    add_definitions("/EHsc")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_COMPILER g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()

# build type
if(NOT CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX d)
endif()
set(CMAKE_CONFIG_DEBUG   DEBUG)
set(CMAKE_CONFIG_RELEASE RELEASE)
set(CMAKE_CONFIG $<$<CONFIG:Debug>:${CMAKE_CONFIG_DEBUG}>$<$<CONFIG:Release>:${CMAKE_CONFIG_RELEASE}>)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

# Offer the user the choice of overriding the installation directories
set(DEFAULT_INSTALL_LIBRARY_DIR lib)
set(DEFAULT_INSTALL_BINARY_DIR  bin)
if(WIN32 AND NOT CYGWIN)
  set(DEFAULT_INSTALL_INCLUDE_DIR include)
  set(DEFAULT_INSTALL_CMAKE_DIR   cmake)
else()
  set(DEFAULT_INSTALL_INCLUDE_DIR include/${cvLeafRecog_PROJECT})
  set(DEFAULT_INSTALL_CMAKE_DIR   lib/cmake/${cvLeafRecog_PROJECT})
endif()
set(INSTALL_LIBRARY_DIR ${DEFAULT_INSTALL_LIBRARY_DIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BINARY_DIR  ${DEFAULT_INSTALL_BINARY_DIR}  CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR ${DEFAULT_INSTALL_INCLUDE_DIR} CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKE_DIR   ${DEFAULT_INSTALL_CMAKE_DIR}   CACHE PATH "Installation directory for CMake files")

# set cvLeafRecog paths
set(cvLeafRecog_ROOT_DIR        "${CMAKE_CURRENT_LIST_DIR}")
set(cvLeafRecog_EXECUTABLE_DIR  "${cvLeafRecog_ROOT_DIR}/bld")
set(cvLeafRecog_SOURCES_DIRS    "${cvLeafRecog_ROOT_DIR}/src")
set(cvLeafRecog_INCLUDE_DIRS    "${cvLeafRecog_ROOT_DIR}/inc")
set(cvLeafRecog_LIBRARY_DIR     "${cvLeafRecog_ROOT_DIR}/lib")
set(CMAKE_INSTALL_PREFIX        "${cvLeafRecog_ROOT_DIR}/install" CACHE STRING "")

# Make relative paths absolute (needed later on)
foreach(p LIBRARY BINARY INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

# =========================
# find header/source files
# =========================

# common files
set(cvLeafRecog_HEADER_FILES ${cvLeafRecog_HEADER_FILES} ${cvLeafRecog_INCLUDE_DIRS}/class_labels.hpp)
set(cvLeafRecog_SOURCE_FILES ${cvLeafRecog_SOURCE_FILES} ${cvLeafRecog_SOURCES_DIRS}/class_labels.cpp)

# LeafNet
set(cvLeafNet_SAMPLE_NAME "LeafNet_predict")
set(cvLeafnet_SOURCE_FILES ${cvLeafnet_SOURCE_FILES} ${cvLeafRecog_SOURCES_DIRS}/${cvLeafNet_SAMPLE_NAME}.cpp)

# Deep-Plant
set(cvDeepPlant_SAMPLE_NAME "DeepPlant_predict")
set(cvDeepPlant_SOURCE_FILES ${cvDeepPlant_SOURCE_FILES} ${cvLeafRecog_SOURCES_DIRS}/${cvDeepPlant_SAMPLE_NAME}.cpp)

# project samples
set(cvLeafRecog_SAMPLES ${cvLeafNet_SAMPLE_NAME} ${cvDeepPlant_SAMPLE_NAME})

# cvLeafRecog configs
set(cvLeafRecog_INCLUDE_DIR       ${cvLeafRecog_INCLUDE_DIRS})
set(cvLeafRecog_LIBRARY_DEBUG     ${cvLeafRecog_PROJECT}${CMAKE_DEBUG_POSTFIX}${CMAKE_${cvLeafRecog_LIBRARY_TYPE}_LIBRARY_SUFFIX})
set(cvLeafRecog_LIBRARY_RELEASE   ${cvLeafRecog_PROJECT}${CMAKE_RELEASE_POSTFIX}${CMAKE_${cvLeafRecog_LIBRARY_TYPE}_LIBRARY_SUFFIX})
set(cvLeafRecog_LIBRARY_DEBUG     ${CMAKE_${cvLeafRecog_LIBRARY_TYPE}_LIBRARY_PREFIX}${cvLeafRecog_LIBRARY_DEBUG})
set(cvLeafRecog_LIBRARY_RELEASE   ${CMAKE_${cvLeafRecog_LIBRARY_TYPE}_LIBRARY_PREFIX}${cvLeafRecog_LIBRARY_RELEASE})

# cvLeafRecog root path (cvLeafRecogOptions)
add_definitions(-DcvLeafRecog_ROOT_PATH="${cvLeafRecog_ROOT_DIR}")

# =========================
# dependencies/packages
# =========================

# find cnpy
set(cvLeafRecog_INCLUDE_DIRS   ${cvLeafRecog_INCLUDE_DIRS}    cnpy)
set(cvLeafRecog_LIBRARIES      ${cvLeafRecog_LIBRARIES}       cnpy)

# find OpenMP
find_package(OpenMP)
if ("${OPENMP_FOUND}" OR "${OPENMP_C_FOUND}" OR "${OPENMP_CXX_FOUND}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(WITH_OPENMP ON)
endif()

# find Boost
##option(Boost_USE_STATIC_LIBS    "Use Boost static libraries"        OFF)
##option(Boost_USE_MULTITHREADED  "Use Boost multithreaded libraries" ON)
##option(Boost_USE_STATIC_RUNTIME "Use Boost static runtime linking"  OFF)
##find_package(Boost REQUIRED COMPONENTS filesystem system)
##if (${Boost_FOUND})
##    set(cvLeafRecog_INCLUDE_DIRS   ${cvLeafRecog_INCLUDE_DIRS}    ${Boost_INCLUDE_DIRS})
##    set(cvLeafRecog_LIBRARIES      ${cvLeafRecog_LIBRARIES}       ${Boost_LIBRARIES})
##    set(cvLeafRecog_LIBRARY_DIRS   ${cvLeafRecog_LIBRARY_DIRS}    ${Boost_LIBRARY_DIRS})
##endif()

# find OpenCV
find_package(OpenCV 3 REQUIRED)
set(cvLeafRecog_INCLUDE_DIRS   ${cvLeafRecog_INCLUDE_DIRS}    ${OpenCV_INCLUDE_DIRS})
set(cvLeafRecog_LIBRARIES      ${cvLeafRecog_LIBRARIES}       ${OpenCV_LIBRARIES})

# =========================
# setup links/targets
# =========================

# include/linking
add_definitions(${cvLeafRecog_LIBRARY_DEFINITION})
add_executable(${cvLeafNet_SAMPLE_NAME}    ${cvLeafRecog_LIBRARY_TYPE} ${cvLeafRecog_SOURCE_FILES}
               ${cvLeafnet_SOURCE_FILES}   ${cvLeafRecog_HEADER_FILES})
add_executable(${cvDeepPlant_SAMPLE_NAME}  ${cvLeafRecog_LIBRARY_TYPE} ${cvLeafRecog_SOURCE_FILES}
               ${cvDeepPlant_SOURCE_FILES} ${cvLeafRecog_HEADER_FILES})
foreach(sample ${cvLeafRecog_SAMPLES})
    if(MSVC)
        set_target_properties(${sample} PROPERTIES LINKER_LANGUAGE C++)
    endif()
    target_include_directories(${sample} PUBLIC ${cvLeafRecog_INCLUDE_DIRS})
    target_link_libraries(${sample} ${cvLeafRecog_LIBRARIES})
endforeach()
link_directories(${cvLeafRecog_LIBRARY_DIRS})

# =========================
# config & install
# =========================

# fix config paths as required
string(REGEX REPLACE "\\\\" "/" INSTALL_INCLUDE_DIR ${INSTALL_INCLUDE_DIR})
string(REGEX REPLACE "\\\\" "/" INSTALL_BINARY_DIR  ${INSTALL_BINARY_DIR})
string(REGEX REPLACE "\\\\" "/" INSTALL_LIBRARY_DIR ${INSTALL_LIBRARY_DIR})

# update config file
##set(cvLeafRecog_CONFIG_INC_DIR ${INSTALL_INCLUDE_DIR})
##set(cvLeafRecog_CONFIG_BIN_DIR ${INSTALL_BINARY_DIR})
##set(cvLeafRecog_CONFIG_LIB_DIR ${INSTALL_LIBRARY_DIR})
##set(cvLeafRecog_CONFIG_LIB_DBG ${INSTALL_LIBRARY_DIR}/${cvLeafRecog_LIBRARY_DEBUG})
##set(cvLeafRecog_CONFIG_LIB_REL ${INSTALL_LIBRARY_DIR}/${cvLeafRecog_LIBRARY_RELEASE})
##set(cvLeafRecog_CONFIG_LIB_DEF ${cvLeafRecog_LIBRARY_DEFINITION})
##set(cvLeafRecog_CONFIG_CMK_DIR ${INSTALL_CMAKE_DIR})
##set(cvLeafRecog_CONFIG_FILE    ${cvLeafRecog_PROJECT}CoLeafNet_object_detectionnfig.cmake)
##if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
##    set(cvLeafRecog_CONFIG_PATH ${cvLeafRecog_CONFIG_FILE})
##else()
##    set(cvLeafRecog_CONFIG_PATH ${CMAKE_BINARY_DIR}/${cvLeafRecog_CONFIG_FILE})
##endif()
##file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
##configure_file(cmake/${cvLeafRecog_CONFIG_FILE}.in ${cvLeafRecog_CONFIG_PATH} @ONLY)

# add dependency find config files to install
##set(cvLeafRecog_FIND_CONFIG ${cvLeafRecog_FIND_CONFIG} "cmake/FindLibSVM.cmake")
##set(cvLeafRecog_FIND_CONFIG ${cvLeafRecog_FIND_CONFIG} "cmake/FindfeHOG.cmake")
##set(cvLeafRecog_FIND_CONFIG ${cvLeafRecog_FIND_CONFIG} "cmake/Findmvector.cmake")

# install
install(TARGETS ${cvLeafRecog_SAMPLES}
        RUNTIME DESTINATION ${INSTALL_BINARY_DIR}
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR})
install(FILES ${cvLeafRecog_HEADER_FILES} DESTINATION ${INSTALL_INCLUDE_DIR})
##install(FILES ${cvLeafRecog_CONFIG_PATH}  DESTINATION ${INSTALL_CMAKE_DIR})
##install(FILES ${cvLeafRecog_FIND_CONFIG}  DESTINATION ${INSTALL_CMAKE_DIR})

