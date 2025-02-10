#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_MotionManagement_Steering_SteeringWheel. Stores all the properties.
*/
struct VehicleMotionManagementSteeringSteeringWheelData
{
    int32_t m_Angle {0};
    int32_t m_AngleTarget {0};
    int32_t m_AngleTargetMode {0};
    int32_t m_TorqueOffsetTarget {0};
    int32_t m_TorqueOffsetTargetMode {0};
    int32_t m_TorqueTarget {0};
    int32_t m_TorqueTargetMode {0};
    int32_t m_Torque {0};
};

}
}