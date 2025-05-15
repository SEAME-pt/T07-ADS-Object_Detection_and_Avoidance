#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain. Stores all the properties.
*/
struct VehiclePowertrainData
{
    float m_AccumulatedBrakingEnergy {0.0f};
    int32_t m_Range {0};
    int32_t m_TimeRemaining {0};
    Vehicle_Powertrain_Type_ValueEnum m_Type {Vehicle_Powertrain_Type_ValueEnum::COMBUSTION};
    int32_t m_PowerOptimizeLevel {0};
    bool m_IsAutoPowerOptimize {false};
};

}
}