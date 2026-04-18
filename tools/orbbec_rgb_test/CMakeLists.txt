cmake_minimum_required(VERSION 3.16)

project(orbbec_rgb_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORBBEC_SDK_ROOT "$ENV{ORBBEC_SDK_ROOT}" CACHE PATH "Path to the Orbbec SDK root")
if(NOT ORBBEC_SDK_ROOT)
    set(ORBBEC_SDK_ROOT "$ENV{HOME}/orbbec/sdk" CACHE PATH "Path to the Orbbec SDK root" FORCE)
endif()

find_package(OrbbecSDK REQUIRED CONFIG PATHS "${ORBBEC_SDK_ROOT}/lib" NO_DEFAULT_PATH)

add_executable(orbbec_rgb_test main.cpp)
target_link_libraries(orbbec_rgb_test PRIVATE ob::OrbbecSDK)

if(APPLE)
    set_target_properties(orbbec_rgb_test PROPERTIES
        BUILD_RPATH "${ORBBEC_SDK_ROOT}/lib;${ORBBEC_SDK_ROOT}/lib/extensions"
    )
endif()
