#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_Charging_Timer. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryChargingTimerData
{
    Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum m_Mode {Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum::INACTIVE};
    std::string m_Time {std::string()};
};

}
}