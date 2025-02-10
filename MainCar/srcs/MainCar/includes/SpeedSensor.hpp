
#pragma once

#include "ISensor.hpp"
#include <iostream>

class SpeedSensor : public ISensor {
    private:
        float _lastSpeed;

    public:
        SpeedSensor(CANBus& can, uint32_t id);

        void initialize();
        int readData();
        
        const std::string getType() const;
        const float getValue() const;
};