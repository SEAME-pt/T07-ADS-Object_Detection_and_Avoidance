#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_LowVoltageBattery. Stores all the properties.
*/
struct VehicleLowVoltageBatteryData
{
    int32_t m_NominalVoltage {0};
    int32_t m_NominalCapacity {0};
    float m_CurrentVoltage {0.0f};
    float m_CurrentCurrent {0.0f};
};

}
}