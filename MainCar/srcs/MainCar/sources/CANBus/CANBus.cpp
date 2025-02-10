#include "CANBus.hpp"
#include <fcntl.h>  // For fcntl function

// Constructor: Initializes the CAN interface
CANBus::CANBus() {
    _socket_fd = -1;
}

// Constructor: Initializes the CAN interface with specified interface and bitrate
CANBus::CANBus(const std::string& interface, int bitrate) {
    // Create a CAN RAW socket
    _socket_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (_socket_fd < 0) {
        throw std::runtime_error("Failed to create CAN socket");
    }

    // Bind the socket to the CAN interface
    struct ifreq ifr;
    std::strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);
    if (ioctl(_socket_fd, SIOCGIFINDEX, &ifr) < 0) {
        close(_socket_fd);
        throw std::runtime_error("Failed to get interface index for: " + interface);
    }

    struct sockaddr_can addr = {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(_socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(_socket_fd);
        throw std::runtime_error("Failed to bind CAN socket to interface: " + interface);
    }

    // Optionally set bitrate (requires `ip link` configuration, not directly in code)
    std::cout << "Initialized CAN interface: " << interface << " at bitrate: " << bitrate << " bps" << std::endl;

    // Set the socket to non-blocking mode using fcntl
    fcntl(_socket_fd, F_SETFL, O_NONBLOCK);
}

// Destructor: Closes the CAN socket
CANBus::~CANBus() {
    if (_socket_fd >= 0) {
        close(_socket_fd);
    }
}

// Sends a message on the CAN bus with retry logic
bool CANBus::sendMessage(uint32_t id, const std::vector<uint8_t>& data) {
    if (data.size() > 8) {
        std::cerr << "Error: CAN frame data exceeds 8 bytes!" << std::endl;
        return false;
    }

    struct can_frame frame = {};
    frame.can_id = id;
    frame.can_dlc = data.size(); // Data Length Code (number of bytes)

    std::copy(data.begin(), data.end(), frame.data);

    // Try sending the message up to 3 times in case of failure
    int attempts = 3;
    while (attempts--) {
        if (write(_socket_fd, &frame, sizeof(frame)) == sizeof(frame)) {
            return true;  // Successful message sending
        }
        std::cerr << "Error: Failed to send CAN message, retrying..." << std::endl;
    }

    std::cerr << "Error: Failed to send CAN message after multiple attempts!" << std::endl;
    return false;  // Failed after 3 attempts
}

// Receives a message from the CAN bus in non-blocking mode
bool CANBus::receiveMessage(uint32_t& id, std::vector<uint8_t>& data) {
    struct can_frame frame = {};

    // Try reading a CAN frame (non-blocking due to fcntl setting)
    ssize_t nbytes = read(_socket_fd, &frame, sizeof(frame));
    if (nbytes < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No message available, return false to indicate no data
            return false;
        } else {
            std::cerr << "Error: Failed to read CAN message!" << std::endl;
            return false;
        }
    }

    if (nbytes < sizeof(frame)) {
        std::cerr << "Error: Incomplete CAN frame received!" << std::endl;
        return false;
    }

    id = frame.can_id;
    data.assign(frame.data, frame.data + frame.can_dlc);

    return true;  // Successfully received a message
}
