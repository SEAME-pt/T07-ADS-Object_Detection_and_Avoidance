#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_MotionManagement_ElectricAxle_Row1. Stores all the properties.
*/
struct VehicleMotionManagementElectricAxleRow1Data
{
    int32_t m_Torque {0};
    int32_t m_TorqueTarget {0};
    int32_t m_TorqueMaximum {0};
    int32_t m_TorqueMinimum {0};
    int32_t m_TorqueLongTermMaximum {0};
    int32_t m_TorqueLongTermMinimum {0};
    int32_t m_TorqueShortTermMaximum {0};
    int32_t m_TorqueShortTermMinimum {0};
    int32_t m_TorqueMaximumLimit {0};
    int32_t m_TorqueMinimumLimit {0};
    int32_t m_RotationalSpeed {0};
    int32_t m_RotationalSpeedTarget {0};
    int32_t m_RotationalSpeedMaximumLimit {0};
    int32_t m_RotationalSpeedMinimumLimit {0};
};

}
}