# Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Atualiza os pacotes e instala dependências necessárias
RUN apt update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libzmq3-dev \
    build-essential \
    python3-pip \
    libgpiod-dev \
    libi2c-dev \
    git \
    cmake \
    pkg-config

RUN pip3 install Jetson.GPIO
# Instala o cabeçalho do ZeroMQ
RUN git clone https://github.com/zeromq/cppzmq.git /tmp/cppzmq && \
    cp /tmp/cppzmq/zmq.hpp /usr/include/ && \
    rm -rf /tmp/cppzmq

# Configuração do workspace
WORKDIR /workspace

# Comando padrão ao entrar no container
CMD ["/bin/bash"]
