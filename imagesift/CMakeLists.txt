cmake_minimum_required(VERSION 2.8.3)
project(imagesift)

# Use ccache if installed to make it fast to generate object files
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs posedetection_msgs image_transport cv_bridge cmake_modules
  jsk_recognition_utils nodelet jsk_topic_tools)

find_package(OpenCV)

catkin_package(
    CATKIN_DEPENDS roscpp sensor_msgs posedetection_msgs image_transport cv_bridge
    LIBRARIES ${PROJECT_NAME}
    INCLUDE_DIRS include
    DEPENDS OpenCV libsiftfast
)

catkin_python_setup()

set(ENV{PKG_CONFIG_PATH} ${CATKIN_DEVEL_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH})
find_package(PkgConfig REQUIRED)
pkg_check_modules(siftfast libsiftfast REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${siftfast_INCLUDE_DIRS})
link_directories(${siftfast_LIBRARY_DIRS})
link_libraries(${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${siftfast_LIBRARIES})


macro(jsk_feature detector extractor exec_name)
  list(APPEND jsk_exec ${exec_name})
  set(DETECTOR ${detector})
  set(EXTRACTOR ${extractor})
  set(EXEC_NAME ${exec_name})
  configure_file(src/imagefeatures.cpp.in src/${exec_name}.cpp) #${CMAKE_CURRENT_BINARY_DIR}/
  add_executable(${exec_name} src/${exec_name}.cpp)
  set_target_properties(${exec_name} PROPERTIES COMPILE_FLAGS "-msse2 -O3" LINK_FLAGS "-msse2 -O3")
  if($ENV{ROS_DISTRO} STREQUAL "groovy" OR $ENV{ROS_DISTRO} STREQUAL "hydro")
    set_target_properties(${exec_name} PROPERTIES COMPILE_FLAGS "-DOPENCV_NON_FREE")
  endif()
  add_dependencies(${exec_name} posedetection_msgs_generate_messages_cpp libsiftfast)
endmacro(jsk_feature detector extractor exec_name)

message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
if("${OpenCV_VERSION}" VERSION_LESS 3.0.0) # http://stackoverflow.com/questions/31509600/why-so-many-featuredetector-and-descriptorextractor-are-not-supported-in-opencv
jsk_feature("SURF" "SURF" "imagesurf")
jsk_feature("STAR" "SURF" "imagestar")
jsk_feature("SIFT" "SURF" "imagesift_surf")
jsk_feature("SIFT" "SIFT" "imagesift_sift")
jsk_feature("ORB" "ORB" "imageorb")
endif()
jsk_feature("BRISK" "BRISK" "imagebrisk")


jsk_nodelet(src/imagesift.cpp
  "imagesift/ImageSift"
  "imagesift_exec"
  nodelet_sources nodelet_executables)
set_target_properties("imagesift_exec" PROPERTIES OUTPUT_NAME "imagesift")
list(APPEND jsk_exec "imagesift_exec")

add_library(${PROJECT_NAME} SHARED ${nodelet_sources})

add_definitions("-O2 -g")
include_directories(include ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)


install(TARGETS ${PROJECT_NAME} ${jsk_exec}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(FILES nodelet.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  if("$ENV{ROS_DISTRO}" STRGREATER "hydro")
    # FIXME: jsk_tools/test_topic_published.py does not work on hydro travis/jenkins
    # https://github.com/jsk-ros-pkg/jsk_common/pull/1293#issuecomment-164158260
    add_rostest(test/test_imagesift.test)
  endif()
endif()
