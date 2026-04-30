cmake_minimum_required(VERSION 3.24)

project(InterPlanetary3D VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)

FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG v0.1.36
)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)

FetchContent_MakeAvailable(glfw glad glm)

find_package(OpenGL REQUIRED)

add_executable(InterPlanetary3D
  src/main.cpp
)

target_link_libraries(InterPlanetary3D
  PRIVATE
    OpenGL::GL
    glfw
    glad
    glm
)

target_compile_definitions(InterPlanetary3D PRIVATE GL_SILENCE_DEPRECATION)

if(APPLE)
  target_compile_definitions(InterPlanetary3D PRIVATE GL_SILENCE_DEPRECATION)
endif()
