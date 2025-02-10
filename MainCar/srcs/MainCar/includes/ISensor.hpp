#pragma once

#include <iostream>
#include "CANBus.hpp"

class ISensor {
    protected:
        CANBus& canBus;
        uint32_t canId;

    public:
        ISensor(CANBus& can, uint32_t id) : canBus(can), canId(id) {}
        virtual ~ISensor() = default;

        virtual void initialize() = 0;
        virtual int readData() = 0;
        virtual const std::string getType() const = 0;
        virtual const float getValue() const = 0;
};
