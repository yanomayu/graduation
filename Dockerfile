FROM nvidia/cuda
MAINTAINER yanomayu
ENV container docker
 

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3 \
    python-pip \
    git \
    wget \
    libmecab-dev \
    python-wheel \
    python-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


RUN pip install --upgrade pip
RUN pip install --no-cache-dir cupy-cuda90 chainer

RUN pip install nltk progressbar2

#RUN pip install --upgrade tensorflow-gpu
#RUN pip install keras
#aiueo