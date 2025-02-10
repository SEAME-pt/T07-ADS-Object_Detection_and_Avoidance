#include "CANBus.hpp"
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <CAN_INTERFACE> [<ID> <BYTE1> <BYTE2> ...]" << std::endl;
        return 1;
    }

    std::string interface = argv[1];
    uint32_t id = 0x123; // Default CAN ID
    std::vector<uint8_t> data;

    // Parse CAN ID and data from command-line arguments (if provided)
    if (argc > 2) {
        id = std::stoul(argv[2], nullptr, 16); // Convert CAN ID to hexadecimal
        for (int i = 3; i < argc; ++i) {
            data.push_back(static_cast<uint8_t>(std::stoul(argv[i], nullptr, 16)));
        }
    }

    try {
        CANBus canBus(interface, 500000);

        std::cout << "Sending CAN message on interface: " << interface << std::endl;
        if (canBus.sendMessage(id, data)) {
            std::cout << "Message sent successfully! ID: 0x" << std::hex << id << " Data:";
            for (uint8_t byte : data) {
                std::cout << " 0x" << std::hex << static_cast<int>(byte);
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to send message!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
