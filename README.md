

# Setting Up a Cross-Compilation Environment with Docker Compose

This guide provides step-by-step instructions to create a Docker container for cross-compiling a project for a Raspberry Pi, setting up directory bindings for real-time updates, and deploying compiled binaries.

---

## Prerequisites

1. A Raspberry Pi with SSH enabled.
2. Docker and Docker Compose installed on your host machine.
3. Basic knowledge of SSH and Docker commands.
4. The IP address and username for your Raspberry Pi.

---

## Step 1: Creating the Docker Compose Configuration

### Docker Compose Configuration File

Create a `docker-compose.yml` file to define services for the cross-compilation environment:

```yaml
dervices:
  main-car:
    build:
      context: ./srcs/MainCar
    command: /bin/bash
    stdin_open: true
    tty: true
    volumes:
      - type: bind
        source: ./srcs/MainCar
        target: /workspace

  controller:
    build:
      context: ./srcs/Controller
    command: /bin/bash
    stdin_open: true
    tty: true
    volumes:
      - type: bind
        source: ./srcs/Controller
        target: /workspace
    depends_on:
      - main-car
```

### Explanation:

- \`\`: Points to the directory containing the `Dockerfile` for each service.
- \`\`: Uses `bind` mounts to sync files between the host and the container.
- \`\`: Ensures `controller` starts after `main-car`.

---

## Step 2: Creating the Dockerfile for Cross-Compilation

Create a `Dockerfile` inside each service directory (e.g., `srcs/MainCar` or `srcs/Controller`):

```dockerfile
# Use an ARM64 Debian image
FROM arm64v8/debian:buster

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    cmake \
    rsync \
    make \
    git \
    tzdata \
    libsdl2-dev \
    libgtest-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pigpio from source
RUN git clone https://github.com/joan2937/pigpio.git /tmp/pigpio \
    && cd /tmp/pigpio \
    && make \
    && make install \
    && rm -rf /tmp/pigpio

# Set the default working directory
WORKDIR /workspace

CMD ["/bin/bash"]
```

### Explanation:

- **Base Image**: Uses `arm64v8/debian:buster` to simulate the Raspberry Pi environment.
- **Dependencies**: Installs necessary tools and libraries, including `pigpio` from the source.
- **Working Directory**: Sets `/workspace` as the default directory for project files.

---

## Step 3: Building and Running the Containers

### Step 3.1: Enable QEMU for ARM

Run the following command to enable QEMU for ARM emulation:

```bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

### Step 3.2: Build and Start the Containers

Build and start the containers using Docker Compose:

```bash
sudo docker compose up --build
```

### Step 3.3: Verify Directory Bindings

Check if the `/workspace` directory inside the container reflects the files from the host:

```bash
docker exec -it <container_name> ls -la /workspace
```

Replace `<container_name>` with the name of the container (e.g., `maincar_main-car_1`).

---

## Step 4: Cross-Compiling the Project

1. Access the container:

   ```bash
   docker exec -it <container_name> /bin/bash
   ```

2. Inside the container, use the appropriate build commands, such as `make` or `cmake`, to compile the project:

   ```bash
   cd /workspace
   make
   ```

3. Locate the compiled binaries in the `/workspace` directory.

---

## Step 5: Deploying to Raspberry Pi

### Copy the Compiled Binary

Use `scp` to transfer the compiled binary to the Raspberry Pi:

```bash
scp /path/to/binary user@ip:/location-in-rasp
```

Replace:

- `/path/to/binary`: Path to the compiled binary on your host machine.
- `user`: Raspberry Pi username.
- `ip`: Raspberry Pi IP address.
- `/location-in-rasp`: Destination path in rasp

### Run the Binary on Raspberry Pi

SSH into the Raspberry Pi and execute the binary:

```bash
ssh team07@192.168.1.186
chmod +x /home/team07/binary
/home/team07/binary
```

---

## Notes and Troubleshooting

1. **Missing Volumes:** If `/workspace` is empty, ensure the paths in `volumes` use absolute paths on the host machine.
2. **QEMU Issues:** If ARM emulation doesn’t work, rerun:
   ```bash
   docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
   ```
3. **Package Availability:** For missing packages (e.g., `pigpio`), build them from source as shown in the Dockerfile.
4. **File Permissions:** Ensure that the user running Docker has the necessary permissions for the directories being bound.

---

With this setup, you can easily cross-compile projects for your Raspberry Pi while leveraging Docker for consistency and portability.

