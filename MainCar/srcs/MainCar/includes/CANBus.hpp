#pragma once

#include <linux/can.h>      // for struct can_frame
#include <linux/can/raw.h>  // for CAN_RAW
#include <sys/socket.h>     // for socket()
#include <sys/ioctl.h>      // for ioctl() e SIOCGIFINDEX
#include <net/if.h>         // for struct ifreq
#include <cstring>          // for std::strncpy
#include <stdexcept>        // for std::runtime_error
#include <unistd.h>         // for close()
#include <iostream>         // for std::cout e std::cerr
#include <vector>

class CANBus {
private:
    int     _socket_fd;

public:
    CANBus();
    CANBus(const std::string& interface, int bitrate);
    ~CANBus();

    bool sendMessage(uint32_t id, const std::vector<uint8_t>& data);
    bool receiveMessage(uint32_t& id, std::vector<uint8_t>& data);
};
