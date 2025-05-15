#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_MotionManagement_Steering_Axle_Row1. Stores all the properties.
*/
struct VehicleMotionManagementSteeringAxleRow1Data
{
    int32_t m_RackPositionOffsetTarget {0};
    int32_t m_RackPositionOffsetTargetMode {0};
    int32_t m_RackPosition {0};
    int32_t m_RackPositionTarget {0};
    int32_t m_RackPositionTargetMode {0};
};

}
}