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
    explicit SocketClient(const std::string& host = "127.0.0.1", int port = 12345);
    SocketClient(const SocketClient& other);
    auto operator=(const SocketClient& other) -> SocketClient&;
    virtual ~SocketClient();

    // Methods
    auto receiveMlData() -> std::tuple<double, double, double, double>;
    void sendState(const Eigen::Vector3d& state);
};

#endif // SOCKET_CLIENT_HPP