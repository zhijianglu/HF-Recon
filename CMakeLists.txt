cmake_minimum_required(VERSION 3.1)
project( HF_Recon )

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -g -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(OpenCV 4.4 REQUIRED)
find_package(Open3D REQUIRED)
find_package(teaserpp REQUIRED)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/include/alignlib
        ${OpenCV_INCLUDE_DIRS}
        ${Open3D_INCLUDE_DIRS}
        )
#link_directories(${PCL_LIBRARY_DIRS})
link_directories(lib)
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "")

# 添加所有源文件，包括头文件和cpp文件

# 生成动态链接库
#add_library(PoseGraphOpt SHARED src/PoseGraphOpt.cpp)
#target_link_libraries(PoseGraphOpt  ${Open3D_LIBRARIES}  )
#set_target_properties(PoseGraphOpt PROPERTIES
#        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
#        )

set(TEASER_LIB
        teaserpp::teaser_registration
#        teaserpp::teaser_features
        teaserpp::teaser_io OpenMP::OpenMP_CXX
        )
set(CO_LIBS
        ${Open3D_LIBRARIES}
        ${OpenCV_LIBRARIES}
#        ${PCL_LIBRARIES}
        )

#todo ====================================================================================
add_executable(multi_way_align_sim
        app/multi_way_align_sim.cpp
        src/PairWiseAlign.cpp
        src/ICP.cpp
        src/O3dVis.cpp
        src/global.cpp

        )
target_link_libraries(multi_way_align_sim  ${Open3D_LIBRARIES} PoseGraphOpt
        ${OpenCV_LIBRARIES} )

#todo ====================================================================================
add_executable(sim_post_process
        app/sim_post_process.cpp
        src/PairWiseAlign.cpp
        src/ICP.cpp
        src/global.cpp
        src/O3dVis.cpp
        )
target_link_libraries(sim_post_process  ${Open3D_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

#align with teaser
add_executable(teaser_coarse_align app/teaser_coarse_align.cpp
        src/O3dVis.cpp
        src/global.cpp
        )
target_link_libraries(teaser_coarse_align  ${Open3D_LIBRARIES} ${TEASER_LIB}  ${OpenCV_LIBRARIES})


