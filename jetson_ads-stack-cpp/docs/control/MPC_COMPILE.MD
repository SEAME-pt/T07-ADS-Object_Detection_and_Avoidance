# Compile the code with the correct include paths

```bash
g++ -std=c++17 -I/usr/include/cppad -I/usr/include/coin-or -o mpc_socket mpc_socket.cpp -lcppad_lib -lipopt
```

# Compile with the Eigen include path

```bash
g++ -I/usr/include/eigen3 -o mpc mpc.cpp -std=c++17
```