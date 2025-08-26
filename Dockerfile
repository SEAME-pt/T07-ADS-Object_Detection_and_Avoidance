FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LLVM_VERSION=18 \
    GCC_VERSION=12 \
    BAZELISK_URL=https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    BUILDIFIER_URL=https://github.com/bazelbuild/buildtools/releases/latest/download/buildifier-linux-amd64 \
    PIGPIO_URL=https://github.com/joan2937/pigpio \
    HOME=/root

# Replace ubuntu.sources config file
COPY config/ubuntu.sources /tmp/ubuntu.sources
RUN cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak && \
    mv /tmp/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources

# Add arm64 architecture
RUN dpkg --add-architecture arm64

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    vim \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    cmake \
    lcov \
    locales \
    file \
    && rm -rf /var/lib/apt/lists/*

# Install recommended Qt6 dependencies for x86
RUN apt-get update && apt-get install -y \
    libfontconfig1-dev \
    libfreetype-dev \
    libx11-dev \
    libx11-xcb-dev \
    libxcb-cursor-dev \
    libxcb-glx0-dev \
    libxcb-icccm4-dev \
    libxcb-image0-dev \
    libxcb-keysyms1-dev \
    libxcb-randr0-dev \
    libxcb-render-util0-dev \
    libxcb-shape0-dev \
    libxcb-shm0-dev \
    libxcb-sync-dev \
    libxcb-util-dev \
    libxcb-xfixes0-dev \
    libxcb-xinerama0-dev \
    libxcb-xkb-dev \
    libxcb1-dev \
    libxext-dev \
    libxfixes-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install other dependencies
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    && rm -rf /var/lib/apt/lists/*

# Add LLVM repository and install LLVM/Clang tools
RUN wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" | tee /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y \
    llvm-${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev \
    clang-${LLVM_VERSION} \
    clangd-${LLVM_VERSION} \
    clang-tools-${LLVM_VERSION} \
    clang-format-${LLVM_VERSION} \
    clang-tidy-${LLVM_VERSION} \
    lld-${LLVM_VERSION} \
    lldb-${LLVM_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Set up GCC
RUN apt-get update && apt-get install -y \
    gcc-${GCC_VERSION} \
    g++-${GCC_VERSION} \
    gcc-${GCC_VERSION}-aarch64-linux-gnu \
    g++-${GCC_VERSION}-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# Set up alternatives for LLVM and GCC tools
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/llvm-cov llvm-cov /usr/bin/llvm-cov-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/aarch64-linux-gnu-gcc aarch64-linux-gnu-gcc /usr/bin/aarch64-linux-gnu-gcc-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-${GCC_VERSION} 100

# Install Arduino lint, yamllint, and gitlint
RUN curl -fsSL https://raw.githubusercontent.com/arduino/arduino-lint/main/etc/install.sh | sh && \
    pip install --break-system-packages yamllint gitlint

# Install libzmq3-dev for x86 and arm64
RUN echo 'deb http://download.opensuse.org/repositories/network:/messaging:/zeromq:/release-stable/xUbuntu_22.04/ /' | \
    tee /etc/apt/sources.list.d/network:messaging:zeromq:release-stable.list && \
    curl -fsSL https://download.opensuse.org/repositories/network:messaging:zeromq:release-stable/xUbuntu_22.04/Release.key | \
    gpg --dearmor | tee /etc/apt/trusted.gpg.d/network_messaging_zeromq_release-stable.gpg > /dev/null && \
    apt-get update && apt-get install -y \
    libzmq3-dev \
    libzmq3-dev:arm64 \
    && rm -rf /var/lib/apt/lists/*

# Install Qt6 for x86 and arm64
RUN apt-get update && apt-get install -y \
    qt6-base-dev \
    qt6-declarative-dev \
    qml6-* \
    qt6-base-dev:arm64 \
    qt6-declarative-dev:arm64 \
    qml6-*:arm64 \
    && rm -rf /var/lib/apt/lists/*

# Install SDL2 for x86 and arm64
RUN apt-get update && apt-get install -y \
    qemu-user-static \
    libsdl2-dev \
    libsdl2-dev:arm64 \
    && rm -rf /var/lib/apt/lists/*

# BUILD pigpio from source and install for arm64
RUN git clone ${PIGPIO_URL}.git

COPY tools/cross_compilation/pigpio/makefile pigpio
RUN cd pigpio && \
    git checkout v79 && \
    make CROSS_PREFIX=aarch64-linux-gnu && \
    make install && \
    make clean && \
    rm -rf pigpio

RUN find /usr/opt/lib -name "*pigpio*" -exec mv {} /usr/lib/aarch64-linux-gnu/ \;

RUN find /usr/opt/include -name "*pigpio*" -exec mv {} /usr/include/aarch64-linux-gnu/ \;

RUN find /usr/opt/bin -name "*pigpio*" -exec mv {} /usr/aarch64-linux-gnu/bin/ \;

# BUILD pigpio from source and install for x86
COPY tools/cross_compilation/pigpio/makefile pigpio
RUN cd pigpio && \
    git checkout v79 && \
    make CROSS_PREFIX=x86_64-linux-gnu && \
    make install && \
    rm -rf pigpio

RUN find /usr/opt/lib -name "*pigpio*" -exec mv {} /usr/lib/x86_64-linux-gnu/ \;

RUN find /usr/opt/include -name "*pigpio*" -exec mv {} /usr/include/x86_64-linux-gnu/ \;

RUN find /usr/opt/bin -name "*pigpio*" -exec mv {} /usr/x86_64-linux-gnu/bin/ \;

# Install Bazelisk
RUN wget ${BAZELISK_URL} && \
    chmod 755 bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/bin/bazelisk && \
    ln -s /usr/bin/bazelisk /usr/bin/bazel

# Install Buildifier
RUN wget ${BUILDIFIER_URL} && \
    chmod 755 buildifier-linux-amd64 && \
    mv buildifier-linux-amd64 /usr/bin/buildifier

# Add bin paths to PATH
RUN echo 'if [ -d "$HOME/bin" ]; then PATH="$HOME/bin:$PATH"; fi' >> $HOME/.bashrc && \
    echo 'if [ -d "$HOME/usr/bin" ]; then PATH="$HOME/usr/bin:$PATH"; fi' >> $HOME/.bashrc

# Configure Git
RUN git config --global --add safe.directory /Team07