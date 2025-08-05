cmake_minimum_required(VERSION 3.10)

project(prova0)

#################################################################################
# Include directories
#################################################################################
include_directories(${CMAKE_SOURCE_DIR}/include)

#################################################################################
# Find packages
#################################################################################
find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)
    message( STATUS "OPENGL_FOUND = ${OPENGL_FOUND}" )
    message( STATUS "OPENGL_LIBRARIES = ${OPENGL_LIBRARIES}" )
else()
    message(SEND_ERROR "OpenGL not found!")
endif()

find_package(GLUT REQUIRED)
if(GLUT_FOUND)
    message( STATUS "GLUT_FOUND = ${GLUT_FOUND}" )
    message( STATUS "GLUT_LIBRARIES = ${GLUT_LIBRARIES}" )
else()
    message(SEND_ERROR "GLUT not found!")
endif()

find_package(GLEW REQUIRED)
if(GLEW_FOUND)
    message( STATUS "GLEW_FOUND = ${GLEW_FOUND}" )
    message( STATUS "GLEW_LIBRARIES = ${GLEW_LIBRARIES}" )
else()
    message(SEND_ERROR "GLEW not found!")
endif()

#################################################################################
# Custom libraries
#################################################################################
file(GLOB src_cpp_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
message( STATUS "Source files : ${src_cpp_files}")
add_library(src_cpp_lib STATIC ${src_cpp_files})


#################################################################################
# Compilation and Linking
#################################################################################
add_executable(main "${CMAKE_SOURCE_DIR}/src/main.cpp")

target_link_libraries(main PRIVATE src_cpp_lib ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})

