#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_DCDC. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryDcdcData
{
    float m_PowerLoss {0.0f};
    float m_Temperature {0.0f};
};

}
}