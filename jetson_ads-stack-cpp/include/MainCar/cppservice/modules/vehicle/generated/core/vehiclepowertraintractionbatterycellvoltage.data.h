#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_CellVoltage. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryCellVoltageData
{
    float m_Min {0.0f};
    int32_t m_IdMin {0};
    float m_Max {0.0f};
    int32_t m_IdMax {0};
    std::list<float> m_CellVoltages {std::list<float>()};
};

}
}