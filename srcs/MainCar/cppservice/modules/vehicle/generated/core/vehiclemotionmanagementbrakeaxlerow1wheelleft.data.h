#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Left. Stores all the properties.
*/
struct VehicleMotionManagementBrakeAxleRow1WheelLeftData
{
    int32_t m_Torque {0};
    int32_t m_TorqueArbitrated {0};
    int32_t m_TorqueFrictionMaximum {0};
    int32_t m_TorqueFrictionMinimum {0};
    int32_t m_OmegaUpper {0};
    int32_t m_OmegaLower {0};
};

}
}