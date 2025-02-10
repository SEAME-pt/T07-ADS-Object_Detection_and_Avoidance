#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Powertrain_Transmission. Stores all the properties.
*/
struct VehiclePowertrainTransmissionData
{
    Vehicle_Powertrain_Transmission_Type_ValueEnum m_Type {Vehicle_Powertrain_Transmission_Type_ValueEnum::UNKNOWN};
    int32_t m_GearCount {0};
    Vehicle_Powertrain_Transmission_DriveType_ValueEnum m_DriveType {Vehicle_Powertrain_Transmission_DriveType_ValueEnum::UNKNOWN};
    float m_TravelledDistance {0.0f};
    int32_t m_CurrentGear {0};
    int32_t m_SelectedGear {0};
    bool m_IsParkLockEngaged {false};
    bool m_IsLowRangeEngaged {false};
    bool m_IsElectricalPowertrainEngaged {false};
    Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum m_PerformanceMode {Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::NORMAL};
    Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum m_GearChangeMode {Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum::MANUAL};
    float m_Temperature {0.0f};
    float m_ClutchEngagement {0.0f};
    int32_t m_ClutchWear {0};
    float m_DiffLockFrontEngagement {0.0f};
    float m_DiffLockRearEngagement {0.0f};
    float m_TorqueDistribution {0.0f};
};

}
}