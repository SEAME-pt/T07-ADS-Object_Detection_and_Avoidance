# Main Car Server Documentation

## Overview
The `Main Car Server` application is responsible for reading sensor data, processing it, and transmitting the information via **ZeroMQ (ZMQ)** to different subscribers. It collects data from the CAN bus and integrates with the **COVESA Vehicle API** for vehicle state management.

## Features
- **Sensor Data Acquisition**: Reads speed and odometer data from the CAN bus.
- **State Management**: Uses COVESA to track and update vehicle parameters.
- **Controller Integration**: Listens for control states from the controller application.
- **Data Transmission**: Publishes real-time data to subscribers using **ZMQ**.

## Dependencies
To install the required dependencies on your Jetson Nano, run:
```sh
sudo apt update
sudo apt install libzmq3-dev libpigpio-dev
```

## Preparing Jetson Nano for COVESA
Copy the necessary COVESA libraries to the system directory:
```sh
cp tools/libvehicle-api.so /usr/lib
cp tools/libvehicle-core.so /usr/lib
cp tools/libvehicle-implementation.so /usr/lib
```

## Architecture
### **Sensor Data Handling**
- The `SpeedSensor` and `Odometer` classes read data from the CAN bus.
- The **speed value** is processed using a smoothing algorithm to reduce sudden fluctuations.
- The **vehicle state** is updated in the COVESA API.
- Data is published via **ZMQ**.

### **Controller State Publishing**
- The `ControllerSubscriber` listens to commands such as **horn, lights, and brakes**.
- These states are published to the **ZMQ network**.

## How to Build and Run
### **Build on Jetson Nano**
```sh
mkdir build && cd build
cmake ..
make
./MainCarServer
```

### **Cross-Compilation & Deployment**
To cross-compile the application and deploy it to the Jetson Nano:

1. **Start the QEMU emulator and Docker Compose:**
   ```sh
   docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
   docker compose up
   ```

2. **Access the build container:**
   ```sh
   docker exec -it t07-ads_lane-detection-controller-1 /bin/bash
   ```

3. **Build the project inside the container:**
   ```sh
   cd build
   cmake ..
   make
   ```

4. **Copy the executable to the Jetson Nano:**
   ```sh
   scp MainCarServer team07@10.21.221.47:/home/team07
   ```

## Sensor Data Processing
The system reads speed and odometer values, then applies smoothing for more accurate readings. The data flow is as follows:

1. Read speed from **SpeedSensor**.
2. Apply **smoothing algorithm** to filter noise.
3. Update vehicle speed in **COVESA API**.
4. Publish sensor data via **ZMQ**.

## Controller Integration
The system listens to control commands such as:
- **Horn activation**
- **Lights control (low, high, turn signals, emergency, parking)**
- **Brake state updates**

These are processed and published to subscribers.

## Conclusion
The `Main Car Server` is a critical component in managing vehicle telemetry and control data. It efficiently integrates CAN bus sensors, COVESA vehicle management, and ZMQ communication, ensuring real-time updates for other system components.

