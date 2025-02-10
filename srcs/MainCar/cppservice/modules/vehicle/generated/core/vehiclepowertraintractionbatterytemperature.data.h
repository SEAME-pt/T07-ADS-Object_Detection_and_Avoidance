#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_Temperature. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryTemperatureData
{
    float m_Average {0.0f};
    float m_Min {0.0f};
    float m_Max {0.0f};
    std::list<float> m_CellTemperature {std::list<float>()};
};

}
}