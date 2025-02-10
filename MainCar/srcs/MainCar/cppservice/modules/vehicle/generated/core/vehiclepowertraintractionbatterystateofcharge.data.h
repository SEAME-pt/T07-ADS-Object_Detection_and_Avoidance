#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_StateOfCharge. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryStateOfChargeData
{
    float m_Current {0.0f};
    float m_Displayed {0.0f};
    float m_CurrentEnergy {0.0f};
};

}
}