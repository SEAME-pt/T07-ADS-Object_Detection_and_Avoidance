#pragma once

#include <nlohmann/json.hpp>
#include <zmq.hpp>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "vehicle.h"

using json = nlohmann::json;
using namespace Cpp::Vehicle;

class vehicleSensors : public IVehicleSubscriber {
private:
    zmq::socket_t& publisher; // Referência para o socket do ZeroMQ

    // Helper para enviar mensagens via ZeroMQ
    void sendMessage(const std::string& key, const std::string& value) {
        std::ostringstream message;
        message << key << " " << value;

        zmq::message_t zmqMessage(message.str().size());
        memcpy(zmqMessage.data(), message.str().c_str(), message.str().size());
        publisher.send(zmqMessage, zmq::send_flags::none);
    }

public:
    explicit vehicleSensors(zmq::socket_t& pub) : publisher(pub) {}

    void onSpeedChanged(float speed) override {
        std::cout << "Speed changed: " << speed << std::endl;
        sendMessage("speed", std::to_string(speed));
    }

    void onLowVoltageSystemStateChanged(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) override {
        sendMessage("low_voltage_system_state", std::to_string(static_cast<int>(LowVoltageSystemState)));
    }

    void onTraveledDistanceChanged(float TraveledDistance) override {
        sendMessage("battery", std::to_string(TraveledDistance));
    }

    void onTraveledDistanceSinceStartChanged(float TraveledDistanceSinceStart) override {
        sendMessage("traveled_distance_since_start", std::to_string(TraveledDistanceSinceStart));
    }

    void onStartTimeChanged(const std::string& StartTime) override {
        sendMessage("start_time", StartTime);
    }

    void onTripDurationChanged(float TripDuration) override {
        sendMessage("trip_duration", std::to_string(TripDuration));
    }

    void onTripMeterReadingChanged(float TripMeterReading) override {
        sendMessage("trip_meter_reading", std::to_string(TripMeterReading));
    }

    void onIsBrokenDownChanged(bool IsBrokenDown) override {
        sendMessage("is_broken_down", IsBrokenDown ? "true" : "false");
    }

    void onIsMovingChanged(bool IsMoving) override {
        sendMessage("is_moving", IsMoving ? "true" : "false");
    }

    void onAverageSpeedChanged(float AverageSpeed) override {
        sendMessage("average_speed", std::to_string(AverageSpeed));
    }

    void onPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) override {
        sendMessage("power_optimize_level", std::to_string(PowerOptimizeLevel));
    }

    void onIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) override {
        sendMessage("is_auto_power_optimize", IsAutoPowerOptimize ? "true" : "false");
    }
};