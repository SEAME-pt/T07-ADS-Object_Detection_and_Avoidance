#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include "JetCar.hpp"
#include "SpeedSubscriber.hpp"

class DebugTools {
    public:
        static void measureAccelerationTime(JetCar& car, int targetSpeed);
        static void autoTunePID(JetCar& car);
};
