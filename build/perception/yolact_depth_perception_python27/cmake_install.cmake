# Install script for directory: /home/venkat/Documents/projects/Perception-ros-tuggerbot/src/perception/yolact_depth_perception_python27

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/venkat/Documents/projects/Perception-ros-tuggerbot/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/venkat/Documents/projects/Perception-ros-tuggerbot/build/perception/yolact_depth_perception_python27/catkin_generated/installspace/yolact_depth_perception_python27.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolact_depth_perception_python27/cmake" TYPE FILE FILES
    "/home/venkat/Documents/projects/Perception-ros-tuggerbot/build/perception/yolact_depth_perception_python27/catkin_generated/installspace/yolact_depth_perception_python27Config.cmake"
    "/home/venkat/Documents/projects/Perception-ros-tuggerbot/build/perception/yolact_depth_perception_python27/catkin_generated/installspace/yolact_depth_perception_python27Config-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolact_depth_perception_python27" TYPE FILE FILES "/home/venkat/Documents/projects/Perception-ros-tuggerbot/src/perception/yolact_depth_perception_python27/package.xml")
endif()

