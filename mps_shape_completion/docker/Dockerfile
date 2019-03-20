FROM tensorflow/tensorflow:latest-gpu

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116

RUN apt-get -yq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yqq install \
    ros-kinetic-ros-base \
    gcovr \
    ccache && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

RUN mkdir -p ~/catkin_ws/src/
