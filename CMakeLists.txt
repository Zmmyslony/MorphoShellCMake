cmake_minimum_required(VERSION 3.16)
project(Morphoshell)


find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS
        "source_files/*.cpp"
)

file(GLOB HEADER_FILES CONFIGURE_DEPENDS
        "source_files/*.hpp"
        "source_files/*.h"
)

add_executable(Morphoshell ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(Morphoshell PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(Morphoshell PRIVATE Eigen3::Eigen)
