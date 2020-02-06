FROM tensorflow/tensorflow:1.15.0

RUN apt-get -yq install lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get -yq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yqq install \
    ros-melodic-desktop \
    qt5-default \
    gcovr \
    ccache && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

RUN mkdir -p ~/catkin_ws/src/
