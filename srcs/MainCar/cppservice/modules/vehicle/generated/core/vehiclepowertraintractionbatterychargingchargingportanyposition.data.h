#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionData
{
    Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum m_SupportedInletTypes {Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC};
    bool m_IsFlapOpen {false};
    bool m_IsChargingCableConnected {false};
    bool m_IsChargingCableLocked {false};
};

}
}