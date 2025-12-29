cmake_minimum_required(VERSION 3.16)
project(magnet_env_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MAGNET_USE_SDL "Enable SDL2 rendering" ON)

find_package(Threads REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

if(MAGNET_USE_SDL)
  pkg_check_modules(SDL2 sdl2)
  if(SDL2_FOUND)
    add_compile_definitions(MAGNET_USE_SDL)
  else()
    message(WARNING "SDL2 not found, rendering disabled")
    set(MAGNET_USE_SDL OFF)
  endif()
endif()

add_executable(magnet_env_server
  src/main.cpp
  src/server.cpp
  src/env.cpp
  src/physics.cpp
  src/magnet.cpp
  src/render.cpp
)

target_include_directories(magnet_env_server PRIVATE ${ZMQ_INCLUDE_DIRS})
target_link_libraries(magnet_env_server PRIVATE ${ZMQ_LIBRARIES} Threads::Threads)

if(MAGNET_USE_SDL AND SDL2_FOUND)
  target_include_directories(magnet_env_server PRIVATE ${SDL2_INCLUDE_DIRS})
  target_link_libraries(magnet_env_server PRIVATE ${SDL2_LIBRARIES})
endif()
