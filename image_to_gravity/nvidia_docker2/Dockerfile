FROM nvidia/cuda:10.2-devel-ubuntu18.04

########## nvidia-docker2 hooks ##########
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
########## basis ##########
RUN apt-get update && apt-get install -y \
	vim \
	wget \
	unzip \
	git \
	build-essential
######### Python ##########
RUN apt-get update &&\
	apt-get install -y \
		python3.6 \
		python3-pip &&\
	pip3 install \
		tqdm \
		matplotlib \
		torch==1.0.1 torchvision==0.2.2 \
		tensorboardX
######### My package ##########
RUN cd /home &&\
	git clone https://github.com/ozakiryota/test_pytorch
######### initial position ##########
WORKDIR /home
