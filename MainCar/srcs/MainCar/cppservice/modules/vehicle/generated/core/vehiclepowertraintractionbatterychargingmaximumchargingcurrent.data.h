#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentData
{
    float m_DC {0.0f};
    float m_Phase1 {0.0f};
    float m_Phase2 {0.0f};
    float m_Phase3 {0.0f};
};

}
}