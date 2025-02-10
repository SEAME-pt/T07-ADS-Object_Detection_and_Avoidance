#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery_Charging. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryChargingData
{
    int32_t m_ChargeLimit {0};
    bool m_IsCharging {false};
    bool m_IsDischarging {false};
    Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum m_StartStopCharging {Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum::START};
    float m_AveragePower {0.0f};
    float m_MaxPower {0.0f};
    std::string m_EvseId {std::string()};
    float m_PowerLoss {0.0f};
    float m_Temperature {0.0f};
    float m_ChargeRate {0.0f};
    int32_t m_TimeToComplete {0};
};

}
}