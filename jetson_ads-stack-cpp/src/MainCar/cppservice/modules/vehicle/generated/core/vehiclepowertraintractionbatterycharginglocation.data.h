#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_Charging_Location. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryChargingLocationData
{
    double m_Latitude {0.0};
    double m_Longitude {0.0};
    double m_Altitude {0.0};
};

}
}