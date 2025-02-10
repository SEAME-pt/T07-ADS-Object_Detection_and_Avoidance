# Use an ARM64 Debian image
FROM arm64v8/debian:buster

# Install dependencies for Qt build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    make \
    git \
    wget \
    python3 \
    perl \
    ninja-build \
    build-essential \
    libgl1-mesa-dev \
    libxcb1-dev \
    libx11-dev \
    libx11-xcb-dev \
    libxext-dev \
    libxcb-keysyms1-dev \
    libxcb-image0-dev \
    libxcb-shm0-dev \
    libxcb-icccm4-dev \
    libxcb-sync-dev \
    libxcb-xfixes0-dev \
    libxcb-shape0-dev \
    libxcb-randr0-dev \
    libxcb-render-util0-dev \
    libxcb-util0-dev \
    libxcb-xinerama0-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and build Qt 6 from source
RUN git clone --branch v6.5.3 https://code.qt.io/qt/qt5.git /tmp/qt5 \
    && cd /tmp/qt5 \
    && ./init-repository --module-subset=default \
    && mkdir build && cd build \
    && ../configure -opensource -confirm-license -release -opengl es2 -nomake examples -nomake tests \
    && cmake --build . --parallel $(nproc) \
    && make install \
    && rm -rf /tmp/qt5

# Add Qt to PATH
ENV PATH="/usr/local/Qt-6.5.3/bin:$PATH"

# Verify installations
RUN cmake --version && qmake --version

# Set the working directory
WORKDIR /workspace

CMD ["/bin/bash"]

