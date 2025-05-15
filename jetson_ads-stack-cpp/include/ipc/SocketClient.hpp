#ifndef SOCKET_CLIENT_HPP
#define SOCKET_CLIENT_HPP

#include <Eigen/Dense>
#include <string>
#include <tuple>

class SocketClient {
private:
    int sockfd;

public:
    // Canonical form
    SocketClient(const std::string& host = "127.0.0.1", int port = 12345);
    SocketClient(const SocketClient& other);
    SocketClient& operator=(const SocketClient& other);
    virtual ~SocketClient();

    // Methods
    std::tuple<double, double, double, double> receive_ml_data();
    void send_state(const Eigen::Vector3d& state);
};

#endif // SOCKET_CLIENT_HPP