FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        vim-tiny \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace/butd
ENV CAFFE_ROOT=/workspace/butd/caffe
WORKDIR $CAFFE_ROOT

# Build and install caffe
RUN pip install -U pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd ..
RUN make -j"$(nproc)"
RUN make pycaffe

# Build fast rcnn lib
RUN cd /workspace/butd/lib && make  

# Set ENV
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Download pretrained model
RUN cd /workspace/butd && wget https://www.dropbox.com/s/5xethd2nxa8qrnq/resnet101_faster_rcnn_final.caffemodel?dl=1 -O resnet101_faster_rcnn_final.caffemodel
WORKDIR /workspace