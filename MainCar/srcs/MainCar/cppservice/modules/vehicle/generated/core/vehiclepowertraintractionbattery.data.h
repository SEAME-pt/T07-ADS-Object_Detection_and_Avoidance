#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_TractionBattery. Stores all the properties.
*/
struct VehiclePowertrainTractionBatteryData
{
    std::string m_Id {std::string()};
    std::string m_ProductionDate {std::string()};
    std::list<std::string> m_ErrorCodes {std::list<std::string>()};
    bool m_IsPowerConnected {false};
    bool m_IsGroundConnected {false};
    int32_t m_GrossCapacity {0};
    int32_t m_NetCapacity {0};
    float m_StateOfHealth {0.0f};
    int32_t m_NominalVoltage {0};
    int32_t m_MaxVoltage {0};
    float m_CurrentVoltage {0.0f};
    float m_CurrentCurrent {0.0f};
    float m_CurrentPower {0.0f};
    float m_AccumulatedChargedEnergy {0.0f};
    float m_AccumulatedConsumedEnergy {0.0f};
    float m_AccumulatedChargedThroughput {0.0f};
    float m_AccumulatedConsumedThroughput {0.0f};
    float m_PowerLoss {0.0f};
    int32_t m_Range {0};
    int32_t m_TimeRemaining {0};
};

}
}