
#pragma once

#include "ISensor.hpp"
#include <iostream>

class Odometer : public ISensor {
    private:
        float _lastDistance;

    public:
        Odometer(CANBus& can, uint32_t id);

        void initialize();
        int readData();
        
        const std::string getType() const;
        const float getValue() const;
};