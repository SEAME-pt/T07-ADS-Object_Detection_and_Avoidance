# Base image
FROM hmacielp/crosscompile-gpio:latest

# Atualiza os pacotes e instala dependências necessárias
RUN apt update && apt-get install -y \
    libzmq3-dev \
    build-essential \
    python3-pip \
    libgpiod-dev \
    libi2c-dev

RUN pip3 install Jetson.GPIO
# Instala o cabeçalho do ZeroMQ
RUN git clone https://github.com/zeromq/cppzmq.git /tmp/cppzmq && \
    cp /tmp/cppzmq/zmq.hpp /usr/include/ && \
    rm -rf /tmp/cppzmq

# Configuração do workspace
WORKDIR /workspace

# Comando padrão ao entrar no container
CMD ["/bin/bash"]
