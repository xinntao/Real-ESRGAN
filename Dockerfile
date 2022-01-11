FROM nvidia/cuda:11.5.0-devel-ubuntu20.04


RUN apt update && \
#     apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt-get -y install tzdata && \
    apt install wget git unzip build-essential -y

RUN mkdir -p /root/workspace/real-esrgan && \
    cd /root/workspace/real-esrgan && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/realesrgan-ncnn-vulkan-20211212-ubuntu.zip && \
    unzip realesrgan-ncnn-vulkan-20211212-ubuntu.zip && \
    chmod +x realesrgan-ncnn-vulkan && \
    rm realesrgan-ncnn-vulkan-20211212-ubuntu.zip

RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.2.198-focal.list https://packages.lunarg.com/vulkan/1.2.198/lunarg-vulkan-1.2.198-focal.list && \
    apt update && \
    apt install vulkan-sdk -y

RUN echo ""

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && /root/miniconda3/bin/conda init bash

RUN git clone https://github.com/levityai/Real-ESRGAN.git /root/workspace/Real-ESRGAN \
    && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -P /root/workspace/Real-ESRGAN/experiments/pretrained_models \
    && mkdir /root/workspace/Real-ESRGAN/outputs \
    && /root/miniconda3/bin/pip install -r /root/workspace/Real-ESRGAN/requirements.txt \
    && /root/miniconda3/bin/pip install basicsr

WORKDIR /root/workspace
