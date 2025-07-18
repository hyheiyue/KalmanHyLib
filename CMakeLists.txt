cmake_minimum_required(VERSION 3.14)
project(KalmanHyLib LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


add_library(KalmanHyLib INTERFACE)
target_include_directories(KalmanHyLib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


install(TARGETS KalmanHyLib EXPORT KalmanHyLibTargets)
install(DIRECTORY include/ DESTINATION include)


install(EXPORT KalmanHyLibTargets
        FILE KalmanHyLibTargets.cmake
        NAMESPACE KalmanHyLib::
        DESTINATION lib/cmake/KalmanHyLib)


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/KalmanHyLibConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/KalmanHyLibConfigVersion.cmake"
    DESTINATION lib/cmake/KalmanHyLib
)
