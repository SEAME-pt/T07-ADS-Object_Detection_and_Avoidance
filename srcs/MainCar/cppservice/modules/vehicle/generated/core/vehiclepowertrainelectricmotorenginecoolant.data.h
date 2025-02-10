#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_ElectricMotor_EngineCoolant. Stores all the properties.
*/
struct VehiclePowertrainElectricMotorEngineCoolantData
{
    float m_Capacity {0.0f};
    float m_Temperature {0.0f};
    Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum m_Level {Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::CRITICALLY_LOW};
    int32_t m_LifeRemaining {0};
};

}
}