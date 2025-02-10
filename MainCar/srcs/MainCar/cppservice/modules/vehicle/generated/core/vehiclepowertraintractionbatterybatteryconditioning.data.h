#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_BatteryConditioning. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryBatteryConditioningData
{
    bool m_IsActive {false};
    bool m_IsOngoing {false};
    Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum m_RequestedMode {Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum::INACTIVE};
    std::string m_StartTime {std::string()};
    std::string m_TargetTime {std::string()};
    float m_TargetTemperature {0.0f};
};

}
}