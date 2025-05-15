#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle. Stores all the properties.
*/
struct VehicleData
{
    Vehicle_LowVoltageSystemState_ValueEnum m_LowVoltageSystemState {Vehicle_LowVoltageSystemState_ValueEnum::UNDEFINED};
    float m_Speed {0.0f};
    float m_TraveledDistance {0.0f};
    float m_TraveledDistanceSinceStart {0.0f};
    std::string m_StartTime {std::string()};
    float m_TripDuration {0.0f};
    float m_TripMeterReading {0.0f};
    bool m_IsBrokenDown {false};
    bool m_IsMoving {false};
    float m_AverageSpeed {0.0f};
    int32_t m_PowerOptimizeLevel {0};
    bool m_IsAutoPowerOptimize {false};
};

}
}