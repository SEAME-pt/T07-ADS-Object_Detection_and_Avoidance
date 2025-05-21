## Cross compilation

### Project folder location context

The following bash instruction are based at the project folder location where the `docker-compose` file is placed.

1 - To activate qemu (arm emulator):

 ```bash
 docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
 ```

2 -  Launch container

```bash
docker compose up
```

`Bind volumes` are generated where the local folder is the same as in target unit


3 - Compile Controller (ControllerExec)

```bash
docker exec -it \[controller container\] /bin/bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake -DCMAKE_BUILD_TYPE=Release
make
```

4  - Copy binary to target unit

```bash
scp ControllerExec team07@10.21.221.47:/home/team07
```

This last instruction copies the executable binary to the target unit home folder