# SEAME-Course-24-25

## Table of Contents

- [1. Development Environment](#1-development-environment)
  - [1.1. Install Docker](#11-install-docker)
  - [1.2. Build the Docker Image](#12-build-the-docker-image)
  - [1.3. Pull Docker Image Used on CI](#13-pull-docker-image-used-on-ci)
  - [1.4. Run Docker Container](#14-run-docker-container)
  - [1.5. Common Issues](#15-common-issues)
  - [1.6. Copy Files from Docker](#16-copy-files-from-docker)
- [2. Tooling](#2-tooling)
  - [2.1. Bazel](#21-bazel)
  - [2.2. Toolchains And Cross-Compilation](#22-toolchains-and-cross-compilation)
  - [2.3. Clang-Tidy](#23-clang-tidy)
    - [Run Analysis](#231-run-analysis)
    - [Apply Fixes](#232-apply-fixes)
  - [2.4. Clang-Format](#24-clang-format)
    - [Run Analysis](#241-run-analysis)
    - [Apply Fixes](#242-apply-fixes)
  - [2.5. Bazel Buildifier](#25-bazel-buildifier)
    - [How to Run](#251-how-to-run)
  - [2.6. Sanitizers](#26-sanitizers)
  - [2.7. Coverage](#27-coverage)
- [3. Releases](#3-releases)

## 1. Development Environment

The development environment is fully containerized using Docker. Follow the steps below to set up your environment.

### 1.1. Install Docker

Ensure Docker is installed on your system. For Ubuntu, use the following commands:

```bash
sudo apt-get update
sudo apt-get install docker.io
```

### 1.2. Build the Docker Image

Once Docker is installed, build the image with:

```bash
sudo docker build -t <container_name> .
```

#### Example

```bash
sudo docker build -t t7seame/app .
```

Explanation:

- `-t team07_docker`: This flags the built image with the name team07_docker. You can choose a different name if needed.
- `.`: The dot (.) at the end specifies the current directory as the build context. Docker will use the Dockerfile in this directory to build the image.

### 1.3. Pull Docker Image Used on CI

To get the same image used by the github actions, run the following command:

```bash
sudo docker pull t7seame/app
```

### 1.4. Run Docker Container

To run the Docker container with the image and necessary volumes mounted, use the following command:

```bash
sudo docker run -t -i \
-v "$(pwd)":/Team07 \
--net=host \
--env="DISPLAY" \
--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
-it <container_name> bash
```

#### Example

```bash
sudo docker run -t -i \
-v "$(pwd)":/Team07 \
--net=host \
--env="DISPLAY" \
--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
-it t7seame/app bash
```

Explanation of flags used:

- `-v "$(pwd)":/Team07`: Mounts the current directory to /Team07 inside the container.
- `--net=host`: Uses the host network for the container, which is helpful for applications that require direct access to the host network (e.g., for GUI applications).
- `--env="DISPLAY"`: Passes the host’s display environment variable to the container for GUI applications.
- `--volume="$HOME/.Xauthority:/root/.Xauthority:rw"`: Allows access to the X server on the host, which is required for running GUI applications inside the container.
- `-it`: Runs the container interactively with a terminal.

### 1.5. Common Issues

If you encounter an error similar to the following when trying to run Qt applications:

```bash
Authorization required, but no authorization protocol specified
qt.qpa.xcb: could not connect to display :1
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, wayland-egl, vnc, minimal, xcb, vkkhrdisplay, offscreen, wayland, linuxfb, minimalegl.
```

#### **Solution: Allow Docker to Access the X Server**

To resolve this, run the following command on the host machine to grant Docker containers access to your X server:

```bash
xhost +local:docker
```

This command allows local Docker containers to connect to the X server and use graphical interfaces.

### 1.6. Copy Files from Docker

Use the following command to copy files from the docker container to your environment:

```bash
sudo docker cp <container_id>:<input_path> <output_path>
```

To find all the `container id` use the command:

```bash
sudo docker ps
```

## 2. Tooling

This section outlines the key tools integrated into the development environment.

### 2.2. Clang-Tidy

Clang-Tidy is a static analysis tool for C++ code. It helps identify potential issues, enforce coding standards, and suggest improvements to your code. This section describes how to integrate and use Clang-Tidy with your project for both analysis and automatic fixes.

#### 2.2.1. Run Analysis

To run Clang-Tidy analysis install the Clang-Tidy package or run the docker image.

Run Clang-Tidy with the following command:

```bash
clang-tidy <filepath>
```

The checks are define on .clang-tidy file.

#### 2.2.2. Apply Fixes

To automatically apply fixes suggested by Clang-Tidy, follow these steps:

1. Generate compile_commands.json:

    This file contains the necessary compilation information for Clang-Tidy to analyze your project files. Run the following command to generate it:

    ```bash
    bazel run @hedron_compile_commands//:refresh_all
    ```

    This ensures that the compile_commands.json file is up-to-date and reflects the latest build settings.

    If this command fails, clean the cache with `bazel clean` and delete an existing `build` directory before trying again.

2. Run Clang-Tidy Analysis:

    ```bash
    clang-tidy <filepath> -p ./compile_commands.json
    ```

    Use the following command to find all relevant C++ source files in the project.

    ```bash
    find . -iname "*.c" -o -iname "*.cc" -o -iname "*.cpp" -o -iname "*.cxx" -o -iname "*.h" -o -iname "*.hpp" -o -iname "*.hxx" | xargs clang-tidy -p ./compile_commands.json
    ```

3. To automatically apply any fixes, use the following commands:

    ```bash
    clang-tidy <filepath> -fix -fix-errors -p ./compile_commands.json
    ```

   To run for all relevant C++ source files:

    ```bash
    find . -iname "*.c" -o -iname "*.cc" -o -iname "*.cpp" -o -iname "*.cxx" -o -iname "*.h" -o -iname "*.hpp" -o -iname "*.hxx" | xargs clang-tidy -fix --fix-errors -p ./compile_commands.json
    ```

### 2.3. Clang-Format

Clang-Format is a tool that automatically formats C++ source code according to a set of predefined or custom style rules. Below are instructions for running Clang-Format analysis and applying automatic fixes to your code locally.

#### 2.3.1. Run Analysis

To check the formatting of your code without making any changes, use the following command. It will run `clang-format` in "dry-run" mode, which simulates formatting and reports any issues without modifying files.

```bash
clang-format <filepath> --dry-run --Werror
```

To run for all relevant C++ source files:

```bash
find . -iname "*.c" -o -iname "*.cc" -o -iname "*.cpp" -o -iname "*.cxx" -o -iname "*.h" -o -iname "*.hpp" -o -iname "*.hxx" | xargs clang-format --dry-run --Werror
```

- `find .` recursively searches for C++ source files in the current directory and its subdirectories.
- `-iname` "*.cpp" -o -iname "*.h" ... ensures common C++ file extensions are included.
- `xargs clang-format --dry-run --Werror` applies clang-format to the found files, in "dry-run" mode (no changes are made), and treats any formatting violations as errors (--Werror).

#### 2.3.2. Apply Fixes

To automatically fix the formatting of your code based on the predefined style, you can use the following command. It will rewrite the files in-place according to Clang-Format's rules.

```bash
clang-format <filepath> -i
```

To run for all relevant C++ source files:

```bash
find . -iname "*.c" -o -iname "*.cc" -o -iname "*.cpp" -o -iname "*.cxx" -o -iname "*.h" -o -iname "*.hpp" -o -iname "*.hxx" | xargs clang-format -i
```

For intended file:

```bash
clang-format -i
```

## 3. Releases

A new release is automatically created for every push to a tag that matches the pattern `v*`.

To create a tag use the command:

```bash
git tag v<release version>
```

After creating the tag locally, push it to the remote repo.

```bash
git push origin v<release version>
```

The release includes a `tar.gz` archive containing the `binaries` compiled for arm64, defined in `//:release_bins_filegroup` target.
Add a new `cc_binary` to the filegroup to include the binary into the next release.
