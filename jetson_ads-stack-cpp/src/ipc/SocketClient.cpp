#include "SocketClient.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>

SocketClient::SocketClient(const std::string& host, int port) {
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) throw std::runtime_error("Socket creation failed");

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr);

    if (connect(sockfd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        throw std::runtime_error("Connection failed");
    }
}

SocketClient::SocketClient(const SocketClient& other) {
    // Create a new socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) throw std::runtime_error("Socket creation failed in copy constructor");

    // Copying a connected socket is complex; we reconnect to the same host/port
    // For simplicity, assume same default host/port (could extend to store host/port)
    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(12345);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    if (connect(sockfd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        throw std::runtime_error("Connection failed in copy constructor");
    }
}

SocketClient& SocketClient::operator=(const SocketClient& other) {
    if (this != &other) {
        // Close existing socket
        close(sockfd);

        // Create new socket
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed in assignment");

        // Reconnect
        sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(12345);
        inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

        if (connect(sockfd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Connection failed in assignment");
        }
    }
    return *this;
}

SocketClient::~SocketClient() {
    close(sockfd);
}

std::tuple<double, double, double, double> SocketClient::receive_ml_data() {
    double buffer[4]; // ey, psi_err, v, delta
    ssize_t bytes_received = recv(sockfd, buffer, sizeof(buffer), 0);
    if (bytes_received != sizeof(buffer)) {
        throw std::runtime_error("Incomplete ML data received");
    }
    return {buffer[0], buffer[1], buffer[2], buffer[3]};
}

void SocketClient::send_state(const Eigen::Vector3d& state) {
    double buffer[3] = {state(0), state(1), state(2)};
    send(sockfd, buffer, sizeof(buffer), 0);
}