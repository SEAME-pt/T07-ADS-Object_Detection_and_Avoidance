#include "CANBus.hpp"
#include <iostream>
#include <vector>
#include <csignal>
#include <atomic>

std::atomic<bool> running(true);

void signalHandler(int signal) {
    running = false;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <CAN_INTERFACE>" << std::endl;
        return 1;
    }

    std::string interface = argv[1];

    // Register signal handler to gracefully exit
    std::signal(SIGINT, signalHandler);

    try {
        CANBus canBus(interface, 500000);

        std::cout << "Listening for CAN messages on interface: " << interface << std::endl;

        while (running) {
            uint32_t id;
            std::vector<uint8_t> data;

            if (canBus.receiveMessage(id, data)) {
                std::cout << "Received message! ID: 0x" << std::hex << id << " Data:";
                for (uint8_t byte : data) {
                    std::cout << " 0x" << std::hex << static_cast<int>(byte);
                }
                std::cout << std::endl;
                break;
            }
        }

        std::cout << "Exiting..." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
