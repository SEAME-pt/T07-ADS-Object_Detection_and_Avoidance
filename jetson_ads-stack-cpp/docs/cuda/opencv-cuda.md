# Tutorial: Installing OpenCV with CUDA and GTK Support on Jetson Nano

This tutorial guides you through installing OpenCV with CUDA and GTK support on a Jetson Nano running Ubuntu (e.g., JetPack 4.x). This setup is ideal for projects like real-time lane detection with TensorRT and GStreamer, enabling GPU-accelerated processing and GUI display with `cv2.imshow()`.

## Prerequisites
- Jetson Nano with JetPack installed (includes CUDA and cuDNN).
- Internet access for downloading dependencies.
- Basic familiarity with Linux commands.

## Step 1: Install System Dependencies
Update the system and install required libraries for OpenCV, CUDA, GStreamer, and GTK:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libv4l-dev libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev pkg-config  # For GUI support
sudo apt-get install -y libatlas-base-dev gfortran
```

## Step 2: Remove Pre-installed OpenCV

The Jetson Nano comes with OpenCV 4.1.1 pre-installed, which lacks CUDA support and may conflict with our build. Remove it:
```bash
sudo apt-get remove -y libopencv-dev python3-opencv
pip3 uninstall opencv-python opencv-contrib-python -y
sudo find /usr -name "*cv2*" -exec rm -rf {} + 2>/dev/null
```

## Step 3: Download OpenCV Source

Download OpenCV and the contrib modules (e.g., version 4.5.5):
```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.5.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.5.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.5.5 opencv
mv opencv_contrib-4.5.5 opencv_contrib
```

## Step 4: Configure OpenCV with CMake

Create a build directory and configure OpenCV with CUDA and GTK support:

```bash
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=5.3 \  # Jetson Nano Maxwell GPU
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF ..
```

Check the CMake output for:

    CUDA: YES
    NVIDIA CUDA: YES
    GUI: GTK: YES If any are missing, ensure the corresponding dependencies are installed (e.g., libgtk2.0-dev for GTK).

## Step 5: Build and Install OpenCV

Compile and install (this takes ~1-2 hours on the Nano):
```bash
make -j4
sudo make install
sudo ldconfig
```

## Step 6: Set Python Path

The compiled OpenCV Python bindings are installed in /usr/local/lib/python3.6/site-packages. Ensure Python finds them by setting PYTHONPATH:
```bash
echo 'export PYTHONPATH=/usr/local/lib/python3.6/site-packages:/home/<your-username>/.local/lib/python3.6/site-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 7: Verify Installation

Check the OpenCV version and build details:
```bash
python
import cv2
print(cv2.__version__)  # Should show 4.5.5
print(cv2.__file__)     # Should be in /usr/local/lib/python3.6/site-packages
print(cv2.getBuildInformation())  # Look for "NVIDIA CUDA: YES" and "GTK: YES"
```

