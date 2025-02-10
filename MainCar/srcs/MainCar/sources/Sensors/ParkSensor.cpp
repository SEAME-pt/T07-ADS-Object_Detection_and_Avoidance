/**
 * @file ParkSensor.cpp
 * @brief Implementation of the ParkSensor class that reads parking sensor data from a CAN bus.
 */

#include "ParkSensor.hpp"

/**
 * @brief Constructs a ParkSensor object.
 *
 * @param can Reference to a CANBus object.
 * @param id The CAN identifier for the park sensor.
 */
ParkSensor::ParkSensor(CANBus& can, uint32_t id) : ISensor(can, id), _distance(0) {
    // Constructor initializes the distance to 0
}

/**
 * @brief Initializes the park sensor.
 */
void ParkSensor::initialize() {
    // Output initialization message to the console
    std::cout << "Park Sensor" << std::endl;
}

/**
 * @brief Reads data from the CAN bus for the park sensor.
 *
 * @return 0 if data is successfully read and processed, -1 otherwise.
 */
int ParkSensor::readData() {
    uint32_t id; // Variable to hold the CAN message ID
    std::vector<uint8_t> data; // Vector to hold the CAN message data

    // Check if a message is received from the CAN bus
    if (canBus.receiveMessage(id, data)) {
        // Check if the received message ID matches the sensor's CAN ID
        if (id == canId) {
            // Output the received message ID and data to the console
            std::cout << "Message received with ID: 0x" 
                      << std::hex << id << std::dec << "\nData: ";

            // Print each byte of the received data
            for (size_t i = 0; i < data.size(); ++i) {
                std::cout << "0x" << std::hex << (int)data[i] << " ";
            }

            std::cout << std::dec << std::endl;

            // Ensure there are at least 2 bytes of data to interpret the distance value
            if (data.size() >= 2) {
                // Combine the two bytes to form the distance value in centimeters
                uint16_t distanceValue = (data[0] << 8) | data[1]; // Little-endian: high byte | low byte

                // Assign the distance value to the _distance variable
                _distance = static_cast<int>(distanceValue); // Ensure it is treated as an integer

                // Output the distance value to the console
                std::cout << "Distance read: " << _distance << " cm" << std::endl;

                return 0; // Indicate success
            } else {
                // Error: Not enough data to interpret the distance value
                std::cerr << "Error: Insufficient data to interpret the distance value." << std::endl;
                return -1; // Indicate failure
            }
        } else {
            // Ignore messages with unexpected IDs
            std::cout << "Message ignored with unexpected ID: 0x" 
                      << std::hex << id << std::dec << std::endl;
            return -1; // Indicate failure
        }
    } else {
        // No message available on the CAN bus
        std::cout << "No message available on the CAN bus." << std::endl;
        return -1; // Indicate failure
    }
}

/**
 * @brief Gets the last distance value read by the sensor.
 *
 * @return The last distance value in centimeters.
 */
const float ParkSensor::getValue() const {
    return _distance;
}

/**
 * @brief Gets the type of the sensor as a string.
 *
 * @return The sensor type, "ParkSensor".
 */
const std::string ParkSensor::getType() const {
    return "ParkSensor";
}
