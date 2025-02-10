FROM hmacielp/crosscompile-covesa

RUN apt update && apt-get install -y \
    libzmq3-dev \
    build-essential

# Instalar a versão mais recente do cppzmq
RUN git clone https://github.com/zeromq/cppzmq.git /tmp/cppzmq && \
    cp /tmp/cppzmq/zmq.hpp /usr/include/ && \
    rm -rf /tmp/cppzmq

WORKDIR /workspace

CMD ["/bin/bash"]
