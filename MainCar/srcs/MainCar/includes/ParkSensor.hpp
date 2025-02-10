
#pragma once

#include "ISensor.hpp"
#include <iostream>

class ParkSensor : public ISensor {
    private:
        int _distance;

    public:
        ParkSensor(CANBus& can, uint32_t id);

        void initialize();
        int readData();
        
        const std::string getType() const;
        const float getValue() const;
};