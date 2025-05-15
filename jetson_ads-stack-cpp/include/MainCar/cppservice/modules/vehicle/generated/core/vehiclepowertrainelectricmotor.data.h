#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_ElectricMotor. Stores all the properties.
*/
struct VehiclePowertrainElectricMotorData
{
    std::string m_EngineCode {std::string()};
    int32_t m_MaxPower {0};
    int32_t m_MaxTorque {0};
    int32_t m_MaxRegenPower {0};
    int32_t m_MaxRegenTorque {0};
    int32_t m_Speed {0};
    float m_TimeInUse {0.0f};
    float m_Temperature {0.0f};
    float m_CoolantTemperature_Deprecated {0.0f};
    int32_t m_Power {0};
    int32_t m_Torque {0};
};

}
}