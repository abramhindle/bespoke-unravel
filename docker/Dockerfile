# This is a comment
FROM ubuntu:16.04
MAINTAINER Abram Hindle <abram.hindle+unravel@ualberta.ca>
RUN apt -qq  update
RUN apt -y install g++ vim flex subversion git unzip libncurses5-dev xz-utils make wget libssl-dev libbz2-dev libz-dev libc6-i386
RUN apt -y install wget make
ADD . /root/
SHELL ["/bin/bash", "-c"]
# RUN (cd /root && git clone https://github.com/abramhindle/setupenv
# setupenv is a submodule here it should be added by ADD
RUN (cd /root/setupenv && bash -x install)
RUN (cd /root/setupenv && source ./setupenv `./latest`)
RUN (cd /root/unravel-0.2/ && ./autogen.sh && cd codec && make && cd .. && make )

