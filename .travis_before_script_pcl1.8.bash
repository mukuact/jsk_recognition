#!/usr/bin/env bash

######################################################
# Install PCL 1.8
######################################################
cd /tmp

version="1.8.0rc2"
url="https://github.com/PointCloudLibrary/pcl/archive/pcl-${version}.tar.gz"
fname=pcl-${version}.tar.gz

wget $url -O $fname
tar zxf $fname

cd pcl-pcl-${version}
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
sudo make -j2 install


######################################################
# Setup dependencies to rebuild from source
######################################################
sudo -H pip install -q rosinstall_generator

rosinstall_generator --tar --rosdistro $ROS_DISTRO \
  pcl_ros \
  octomap_server \
>> /tmp/$$.rosinstall

cd ~/ros/ws_$REPOSITORY_NAME/src
wstool merge /tmp/$$.rosinstall
wstool up -j4
